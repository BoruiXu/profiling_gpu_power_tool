cmake_minimum_required(VERSION 3.14.0)
project(dcgm_sdk_samples)
find_package(dcgm)


add_executable(power_profiling)
target_sources(power_profiling PRIVATE power_profiling.cpp)
target_link_libraries(power_profiling PRIVATE $<BUILD_INTERFACE:dcgm> $<INSTALL_INTERFACE:Dcgm::dcgm>)