cmake_minimum_required(VERSION 3.15)
project(untitled21)

set(CMAKE_CXX_STANDARD 17)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

add_executable(untitled21 main_1.cpp)
