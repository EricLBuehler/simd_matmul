cmake_minimum_required(VERSION 3.24)
project(simd_matmul)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (CMAKE_BUILD_TYPE STREQUAL "DBG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif()

include_directories(numcpp "/home/ericbuehler/NumCpp/include")

add_executable(simd_matmul src/main.cpp)