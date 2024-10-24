// Borui Xu
//This file is for power log collection using DCGM API

#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "string.h"
#include <iostream>

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
struct power_log{

    double* power_array;
    double* instant_power_array;  
    long long* timestamp_array;
    long long* energy_array;
    long long* frequency_array;
    size_t index = 0;
    //possible array for temperature, frequency, etc.  
    

    //constructor
    power_log(size_t size){
        power_array = new double[size];
        timestamp_array = new long long[size];
        energy_array = new long long[size];
    }

    ~power_log(){
        delete[] power_array;
        delete[] timestamp_array;
        delete[] energy_array;
    }

    //define a function to print the power and timestamp
    //if the index is not 0, print the time time interval
    void print_power_log(){
        for(size_t i = 0; i < index; i++){
            if(i==0)
                std::cout << "Power: " << power_array[i] << " Timestamp: " << timestamp_array[i] << std::endl;
            else
                std::cout << "Power: " << power_array[i] << " Timestamp: " << power_array[i] << " Time interval: " << timestamp_array[i] - timestamp_array[i-1] << std::endl;
        }
    }


    void save_power_log(const std::string &filename) {
        // 打开输出文件
        std::ofstream outFile(filename);
        if (!outFile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // 写入数据到文件，格式为：时间，功耗
        for (size_t i = 0; i < index; ++i) {
            outFile << timestamp_array[i] << ", " << std::fixed << std::setprecision(2) << energy_array[i] << std::endl;
        }

        // 关闭文件
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
    size_t profiling_time_length = 100;//sec
    size_t time_interval = 100000; //usec


    long long tmp_time;  // usec

    //power_log instance
    power_log power_log_instance(20*1000);
    power_log* power_log_instance_ptr = &power_log_instance;

    int python_result;
    
    std::cout << "Embedded mode selected.\n";


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

    // Check the result to see if our DCGM operation was successful.
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

    // add gpu 2
    result = dcgmGroupAddDevice(dcgmHandle, myGroupId, 2);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error adding device to group. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "Successfully added GPU " << 1 << " to group.\n";
    }


   
    dcgmUpdateAllFields(dcgmHandle, 0);
    


    healthSystems = (dcgmHealthSystems_t)(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);

    result = dcgmHealthSet(dcgmHandle, myGroupId, healthSystems);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting health systems. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    //log power temperature frequency and others
    dcgmFieldGrp_t fieldGroupId;
    unsigned short fieldIds[fieldId_size];
    // DCGM_FI_DEV_POWER_USAGE_INSTANT,DCGM_FI_DEV_GPU_TEMP, DCGM_FI_DEV_POWER_USAGE;
    //DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    for(int i = 0; i < fieldId_size; i++){
        fieldIds[i] = DCGM_FI_DEV_POWER_USAGE_INSTANT;
    }
    

    result = dcgmFieldGroupCreate(dcgmHandle, fieldId_size, &fieldIds[0], (char *)"interesting_fields", &fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating field group. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    result = dcgmWatchFields(dcgmHandle, myGroupId, fieldGroupId, time_interval, profiling_time_length, collection_numer);

    // Check result to see if DCGM operation was successful.
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
    
    python_result = system("python api_test.py");

    //sleep 20sec
    // std::this_thread::sleep_for(std::chrono::seconds(20));



   
    dcgmUpdateAllFields(dcgmHandle, 0);
    
    // std::cout << "sleep 20sec.\n";

    results.version = dcgmHealthResponse_version4;

    result = dcgmHealthCheck(dcgmHandle, myGroupId, (dcgmHealthResponse_t *)&results);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error checking health systems. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // Let's display any errors caught by the health watches.
    if (results.overallHealth == DCGM_HEALTH_RESULT_PASS)
    {
        std::cout << "Group is healthy.\n";
    }
    else
    {
        std::cout << "Group has a "
                  << ((results.overallHealth == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");
        std::cout << "GPU ID : Health \n";
        for (unsigned int i = 0; i < results.incidentCount; i++)
        {
            if (results.incidents[i].entityInfo.entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            std::cout << results.incidents[i].entityInfo.entityId << " : ";

            switch (results.incidents[i].health)
            {
                case DCGM_HEALTH_RESULT_PASS:
                    std::cout << "Pass\n";
                    break;
                case DCGM_HEALTH_RESULT_WARN:
                    std::cout << "Warn\n";
                    break;
                default:
                    std::cout << "Fail\n";
            }


            // A more in depth case check may be required here, but since we are only interested in PCIe and memory
            // watches This is all we are going to check for here.
            std::cout << "Error: " << ((results.incidents[i].system == DCGM_HEALTH_WATCH_PCIE) ? "PCIe " : "Memory ");
            std::cout << "watches detected a "
                      << ((results.incidents[i].health == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");

            std::cout << results.incidents[i].error.msg << "\n";
        }
        std::cout << std::endl;
    }

    // And let's also display some of the samples recorded by our watches on the temperature and power usage of the
    // GPUs. We demonstrate how we can pass through information using the userDate void pointer parameter in this
    // function by passing our testString through. More information on this can be seen in the function below.

    //test output
    // for(int i =0;i<3;i++){
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //     //result = dcgmGetLatestValues(dcgmHandle, myGroupId, fieldGroupId, &displayFieldValue, (void *)myTestString);
    // }
    


    //test dcgmGetValuesSince
    result = dcgmGetValuesSince(dcgmHandle, myGroupId, fieldGroupId, (long long )(0),&tmp_time,&displayFieldValue, (void *)power_log_instance_ptr);

    
    // power_log_instance.print_power_log();
    power_log_instance.save_power_log("fintuning_power_log.log");

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error fetching latest values for watches. Result: " << errorString(result) << std::endl;
        goto cleanup;
    }
    

// Cleanup consists of destroying our group and shutting down DCGM.
cleanup:
    std::cout << "Cleaning up. \n";
    dcgmFieldGroupDestroy(dcgmHandle, fieldGroupId);
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    
    dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}


// In this function we simply print out the information in the callback of the watched field.
int displayFieldValue(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData)
{
    
    //convert the void pointer to power_log pointer
    power_log* power_log_instance_ptr = (power_log*)userData;


    // Output the information to screen.
    for (int i = 0; i < numValues; i++)
    {

        size_t current_index = power_log_instance_ptr->index;

        power_log_instance_ptr->power_array[current_index] = values[i].value.dbl;
        // power_log_instance_ptr->energy_array[current_index] = values[i].value.i64;
        power_log_instance_ptr->timestamp_array[current_index] = values[i].ts; 

        power_log_instance_ptr->index++;
        // switch (DcgmFieldGetById(values[i].fieldId)->fieldType)
        // {
        //     case DCGM_FT_BINARY:
        //         // Handle binary data
        //         break;
        //     case DCGM_FT_DOUBLE:
        //         std::cout << "Value: " << values[i].value.dbl;
        //         break;
        //     case DCGM_FT_INT64:
        //         std::cout << "Value: " << values[i].value.i64;
        //         break;
        //     case DCGM_FT_STRING:
        //         std::cout << "Value: " << values[i].value.str;
        //         break;
        //     case DCGM_FT_TIMESTAMP:
        //         std::cout << "Value: " << values[i].value.i64;
        //         break;
        //     default:
        //         std::cout << "Error in field types. " << values[i].fieldType << " Exiting.\n";
        //         // Error, return > 0 error code.
        //         return 1;
        // }
    }

   
    return 0;
}
