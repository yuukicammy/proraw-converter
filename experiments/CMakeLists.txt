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

find_package(xtensor REQUIRED)
if(NOT ${XTENSOR_FOUND})
    message(FATAL_ERROR "** Unable to locate Xtensor.")
else()
    message(STATUS "** Using Xtensor")
endif()

find_package(xtensor-blas REQUIRED)

find_package(Boost COMPONENTS log log_setup REQUIRED)
add_definitions("-DBOOST_LOG_DYN_LINK")

add_executable(my_conversion my_conversion.cpp)
target_compile_definitions(my_conversion PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(my_conversion PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(my_conversion PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  

add_executable(libraw_conversion libraw_conversion.cpp)
target_compile_definitions(libraw_conversion PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(libraw_conversion PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(libraw_conversion PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  

add_executable(xyz_adjustment xyz_adjustment.cpp)
target_compile_definitions(xyz_adjustment PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(xyz_adjustment PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(xyz_adjustment PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  

add_executable(effect_check effect_check.cpp)
target_compile_definitions(effect_check PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(effect_check PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(effect_check PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  

add_executable(save_prophoto16 save_prophoto16.cpp)
target_compile_definitions(save_prophoto16 PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(save_prophoto16 PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/rawconverter/include ${PROJECT_SOURCE_DIR}/third_party ${Boost_INCLUDE_DIR}) 
target_link_libraries(save_prophoto16 PRIVATE ${LibRaw_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})  


set(CMAKE_CXX_STANDARD 17)

