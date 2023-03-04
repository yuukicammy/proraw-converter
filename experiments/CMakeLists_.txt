find_package(LibRaw ${LIBRAW_MIN_VERSION} REQUIRED)
if(NOT ${LIBRAW_FOUND})
    message(FATAL_ERROR "** Unable to locate LibRaw.")
endif()

find_package(OpenCV REQUIRED) 
if(NOT ${OPENCV_FOUND})
    message(FATAL_ERROR "** Unable to locate OpenCV.")
else()
    message(STATUS "** Using OpenCV ${OpenCV_Version}")
endif()

find_package(Boost COMPONENTS log log_setup REQUIRED)
add_definitions("-DBOOST_LOG_DYN_LINK")

add_executable(save_prophoto16 save_prophoto16.cpp)
target_compile_definitions(save_prophoto16 PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(save_prophoto16 PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(save_prophoto16 PRIVATE ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  

set(CMAKE_CXX_STANDARD 17)

