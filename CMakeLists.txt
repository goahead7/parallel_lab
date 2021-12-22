cmake_minimum_required(VERSION 3.17)
project(parallel_final LANGUAGES CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(parallel_final main.cpp src/threads_num.cpp src/functions.cpp src/integrate_cpp.cpp src/integrate_omp.cpp src/reduction.cpp src/types.h)
target_link_libraries(parallel_final PRIVATE OpenMP::OpenMP_CXX)