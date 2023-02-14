set(CMAKE_MIN_VERSION   "3.5.0")
set(LIBRAW_MIN_VERSION  "0.18")
cmake_minimum_required(VERSION ${CMAKE_MIN_VERSION})

if(NOT DEFINED CMAKE_DEBUG_POSTFIX) 
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

set(CMAKE_CXX_STANDARD 17)

project(ProRawConverter)

message(STATUS "----------------------------------------------------------------------------------")
message(STATUS "Starting CMake configuration for: ${PROJECT_NAME}")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH} )

add_subdirectory(experiments)
add_subdirectory(test)