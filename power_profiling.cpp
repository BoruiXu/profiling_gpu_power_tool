// Borui Xu
//This file is for power log collection using DCGM API

#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "string.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>


#include <chrono>
#include <thread>
#include <fstream>  // 用于文件操作
#include <iomanip>  // 用于设置输出格式
#include <cstdlib>


// See function description at bottom of file.
int displayFieldValue(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData);

bool ReceivedIncompatibleMigConfigurationMessage(dcgmDiagResponse_v10 &response)
{
    return (strstr(response.systemError.msg, "MIG configuration is incompatible with the diagnostic")
            || strstr(response.systemError.msg,
                      "Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode"));
}



//a tool to profilr the GPU metircs like power, temperature, frequency, etc.
// need some customization for different metrics
// profiling GPU index
// profiling time length
// time interval
// profiling program command


// currently only support 1 metric
struct power_log {
    long long* timestamp_array;
    double* instant_power_array;
    double* power_array;  
    long long* energy_array;
    size_t* frequency_array;

    size_t index = 0;
    size_t metric_type = 0; // Store the metric type for saving data
    size_t gpu_index = 0;
    size_t total_gpu = 0;

    // Constructor
    power_log(size_t size, size_t metric_type,size_t gpu_index, size_t total_gpu) : 
            metric_type(metric_type), gpu_index(gpu_index), total_gpu(total_gpu), index(0) {
        timestamp_array = new long long[size];

        switch (metric_type) {
            case 1:
                instant_power_array = new double[size];
                break;
            case 2:
                power_array = new double[size];
                break;
            case 3:
                energy_array = new long long[size];
                break;
            case 4:
                frequency_array = new size_t[size];
                break;
            default:
                throw std::invalid_argument("Invalid metric type");
        }
    }

    // Destructor
    ~power_log() {
        delete[] timestamp_array;
        if (metric_type == 1) {
            delete[] instant_power_array;
        } else if (metric_type == 2) {
            delete[] power_array;
        } else if (metric_type == 3) {
            delete[] energy_array;
        } else if (metric_type == 4) {
            delete[] frequency_array;
        }
    }

    void save_power_log(const std::string &filename) {
        // Open output file
        std::ofstream outFile(filename);
        if (!outFile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Write data to file based on metric type
        for (size_t i = 0; i < index; ++i) {
            outFile << timestamp_array[i] << ", ";
            switch (metric_type) {
                case 1:
                    outFile << instant_power_array[i];
                    break;
                case 2:
                    outFile << power_array[i];
                    break;
                case 3:
                    outFile << energy_array[i];
                    break;
                case 4:
                    outFile << frequency_array[i];
                    break;
                default:
                    break; // This case should not occur
            }
            outFile << std::endl;
        }

        // Close file
        outFile.close();
    }
};





int main(int argc, char **argv)
{
    // DCGM calls return a dcgmReturn_t which can be useful for error handling and control flow.
    // Whenever we call DCGM we will store the return in result and check it for errors.
    dcgmReturn_t result;

    //some variables for DCGM
    dcgmHandle_t dcgmHandle = (dcgmHandle_t)NULL;
    dcgmGpuGrp_t myGroupId  = (dcgmGpuGrp_t)NULL;
    char groupName[]        = "myGroup";
    dcgmHealthSystems_t healthSystems;
    dcgmHealthResponse_v4 results;
    dcgmDiagResponse_v10 diagnosticResults;
    const size_t fieldId_size = 1;

    //variable for collection data number
    size_t collection_numer = 0;
    int command_result;
    std::string output_filename;

    //parameters
    size_t profiling_time_length = 100;//sec
    size_t time_interval = 100000; //usec
    size_t metric_type = 1; // Store the metric type for saving data
    std::string command_string; 
    std::string gpu_index_string;


    long long tmp_time;  // usec

    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-t") {
            if (i + 1 < argc) {
                profiling_time_length = std::stoull(argv[++i]);
            } else {
                std::cerr << "Error: -t requires a value." << std::endl;
                return 1;
            }
        } else if (arg == "-i") {
            if (i + 1 < argc) {
                time_interval = std::stoull(argv[++i]);
            } else {
                std::cerr << "Error: -i requires a value." << std::endl;
                return 1;
            }
        } else if (arg == "-m") {
            if (i + 1 < argc) {
                metric_type = std::stoull(argv[++i]);
            } else {
                std::cerr << "Error: -m requires a value." << std::endl;
                return 1;
            }
        } else if (arg == "-g") {
            if (i + 1 < argc) {
                gpu_index_string = argv[++i];
            } else {
                std::cerr << "Error: -g requires a value." << std::endl;
                return 1;
            }
        } else if(arg== "-c"){
            command_string = argv[++i];
        }
        else{
            std::cerr << "Error: Unexpected argument " << arg << std::endl;
            return 1;
        } 
        
    }

    // 处理GPU索引
    std::vector<size_t> gpu_indices;
    std::stringstream ss(gpu_index_string);
    std::string index;

    while (std::getline(ss, index, ',')) {
        gpu_indices.push_back(std::stoull(index)); // 将GPU索引存入vector
    }


    //check must provide GPU index and command
    if (gpu_indices.empty() || command_string.empty()) {
        std::cerr << "Error: Must provide GPU index and command." << std::endl;
        return 1;
    }

    // std::cout << "Profiling Time Length: " << profiling_time_length << std::endl;
    // std::cout << "Time Interval: " << time_interval << std::endl;
    // std::cout << "Metric Type: " << metric_type << std::endl;
    // std::cout << "Command String: " << command_string << std::endl;
    // std::cout << "GPU Indices: ";
    // for (const auto& gpu_index : gpu_indices) {
    //     std::cout << gpu_index << " ";
    // }
    // std::cout << std::endl;


    //power_log array
    power_log* power_log_array[gpu_indices.size()];
    for (int i = 0; i < gpu_indices.size(); i++) {
        power_log_array[i] = new power_log(profiling_time_length*(1000000/time_interval)+100, metric_type, gpu_indices[i], gpu_indices.size());
    }



//     //power_log instance
    // power_log power_log_instance(profiling_time_length*(1000000/time_interval)+100, metric_type, 0, gpu_indices.size());
    // power_log* power_log_instance_ptr = &power_log_instance;

    
    std::cout << "DCGM Embedded mode selected.\n";

    result = dcgmInit();

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error initializing DCGM engine. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "DCGM Initialized.\n";
    }

   
    result = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error starting embedded DCGM engine. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    

        //create a group for gpu 1

    result = dcgmGroupCreate(dcgmHandle, DCGM_GROUP_EMPTY, groupName, &myGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating group. Return: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }
    else
    {
        std::cout << "Successfully created group with group ID: " << (unsigned long)myGroupId << std::endl;
        ;
    }

//     // add gpu index

    // Add the even numbered GPUs
    for (int i = 0; i < gpu_indices.size(); i++)
    {
        
        result = dcgmGroupAddDevice(dcgmHandle, myGroupId, gpu_indices[i]);

        // Check the result to see if our DCGM operation was successful.
        if (result != DCGM_ST_OK)
        {
            std::cout << "Error adding device to group. Return: " << errorString(result) << std::endl;
            goto cleanup;
        }
        else
        {
            std::cout << "Successfully added GPU " << gpu_indices[i] << " to group.\n";
        }
        
    }

   
    dcgmUpdateAllFields(dcgmHandle, 0);
    


//     healthSystems = (dcgmHealthSystems_t)(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);

//     result = dcgmHealthSet(dcgmHandle, myGroupId, healthSystems);

//     // Check result to see if DCGM operation was successful.
//     if (result != DCGM_ST_OK)
//     {
//         std::cout << "Error setting health systems. Result: " << errorString(result) << std::endl;
//         ;
//         goto cleanup;
//     }

    //log power temperature frequency and others
    dcgmFieldGrp_t fieldGroupId;
    unsigned short fieldIds[fieldId_size];
    //https://docs.nvidia.com/datacenter/dcgm/1.6/dcgm-api/group__dcgmFieldIdentifiers.html
    for(int i = 0; i < fieldId_size; i++){
        //according to the metric type
        if(metric_type == 1){
            fieldIds[i] = DCGM_FI_DEV_POWER_USAGE_INSTANT;
        }
        else if(metric_type == 2){
            fieldIds[i] = DCGM_FI_DEV_POWER_USAGE;
        }
        else if(metric_type == 3){
            fieldIds[i] = DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION;
        }
        else if(metric_type == 4){
            fieldIds[i] = DCGM_FI_DEV_SM_CLOCK;
        }
        else{
            std::cerr << "Error: Invalid metric type." << std::endl;
            return 1;
        }
        // fieldIds[i] = DCGM_FI_DEV_POWER_USAGE_INSTANT;
    }
    

    result = dcgmFieldGroupCreate(dcgmHandle, fieldId_size, &fieldIds[0], (char *)"interesting_fields", &fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating field group. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    result = dcgmWatchFields(dcgmHandle, myGroupId, fieldGroupId, time_interval, profiling_time_length, collection_numer);

//     // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting watches. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

   
    dcgmUpdateAllFields(dcgmHandle, 0);
    
    /***********************************************
     *    Run process here while updating fields
     ***********************************************/
    
    command_result = system(command_string.c_str());
   
    dcgmUpdateAllFields(dcgmHandle, 0);
    


    // results.version = dcgmHealthResponse_version4;

    // result = dcgmHealthCheck(dcgmHandle, myGroupId, (dcgmHealthResponse_t *)&results);

//     // Check result to see if DCGM operation was successful.
//     if (result != DCGM_ST_OK)
//     {
//         std::cout << "Error checking health systems. Result: " << errorString(result) << std::endl;
//         ;
//         goto cleanup;
//     }

//     // Let's display any errors caught by the health watches.
//     if (results.overallHealth == DCGM_HEALTH_RESULT_PASS)
//     {
//         std::cout << "Group is healthy.\n";
//     }
//     else
//     {
//         std::cout << "Group has a "
//                   << ((results.overallHealth == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");
//         std::cout << "GPU ID : Health \n";
//         for (unsigned int i = 0; i < results.incidentCount; i++)
//         {
//             if (results.incidents[i].entityInfo.entityGroupId != DCGM_FE_GPU)
//             {
//                 continue;
//             }

//             std::cout << results.incidents[i].entityInfo.entityId << " : ";

//             switch (results.incidents[i].health)
//             {
//                 case DCGM_HEALTH_RESULT_PASS:
//                     std::cout << "Pass\n";
//                     break;
//                 case DCGM_HEALTH_RESULT_WARN:
//                     std::cout << "Warn\n";
//                     break;
//                 default:
//                     std::cout << "Fail\n";
//             }


//             // A more in depth case check may be required here, but since we are only interested in PCIe and memory
//             // watches This is all we are going to check for here.
//             std::cout << "Error: " << ((results.incidents[i].system == DCGM_HEALTH_WATCH_PCIE) ? "PCIe " : "Memory ");
//             std::cout << "watches detected a "
//                       << ((results.incidents[i].health == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");

//             std::cout << results.incidents[i].error.msg << "\n";
//         }
//         std::cout << std::endl;
//     }

//     // And let's also display some of the samples recorded by our watches on the temperature and power usage of the
//     // GPUs. We demonstrate how we can pass through information using the userDate void pointer parameter in this
//     // function by passing our testString through. More information on this can be seen in the function below.

    

    
    //get result dcgmGetValuesSince
    result = dcgmGetValuesSince(dcgmHandle, myGroupId, fieldGroupId, (long long )(0),&tmp_time,&displayFieldValue, (void *)power_log_array);
    // Check result to see if DCGM operation was successful.
    
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error fetching latest values for watches. Result: " << errorString(result) << std::endl;
        goto cleanup;
    }
    
    //define output name according to the metric type
    
    switch (metric_type) {
        case 1:
            output_filename = "instant_power_log.log";
            break;
        case 2:
            output_filename = "power_log.log";
            break;
        case 3:
            output_filename = "energy_log.log";
            break;
        case 4:
            output_filename = "frequency_log.log";
            break;
        default:
            break; // This case should not occur
    }

    // Save power log to file

    for (int i = 0; i < gpu_indices.size(); i++) {
        power_log_array[i]->save_power_log(std::to_string(gpu_indices[i]) + "_" + output_filename);
    }

    //print done
    std::cout << "Profiling done. \n";


// // Cleanup consists of destroying our group and shutting down DCGM.
cleanup:
    std::cout << "Cleaning up. \n";
    dcgmFieldGroupDestroy(dcgmHandle, fieldGroupId);
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    
    dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;

    return 0;
}


// In this function we simply print out the information in the callback of the watched field.
int displayFieldValue(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData)
{
    
    //convert the void pointer to power_log pointer
    
    auto power_log_instance_ptr = static_cast<power_log**>(userData);
    size_t total_gpu = power_log_instance_ptr[1]->total_gpu;
    power_log* current_p;
    size_t metric_type;
    for(int i=0;i<total_gpu;i++){
        if(power_log_instance_ptr[i]->gpu_index == gpuId){
            current_p = power_log_instance_ptr[i];
            metric_type = current_p->metric_type;
            break;
        }
    }
    // std::cout << "GPU ID: " << current_p->gpu_index << std::endl;
    
    

    // Output the information to screen.
    for (int i = 0; i < numValues; i++)
    {

        size_t current_index = current_p->index;
        
        current_p->timestamp_array[current_index] = values[i].ts;
        //according to the metric type
        switch(metric_type){
            case 1:
                current_p->instant_power_array[current_index] = values[i].value.dbl;
                break;
            case 2:
                current_p->power_array[current_index] = values[i].value.dbl;
                break;
            case 3:
                current_p->energy_array[current_index] = values[i].value.i64;
                break;
            case 4:
                current_p->frequency_array[current_index] = values[i].value.i64;
                break;
            default:
                break;
        }
        
        current_p->index++;
        
    }
   
    return 0;
}
