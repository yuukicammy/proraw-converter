find_package(LibRaw ${LIBRAW_MIN_VERSION} REQUIRED)
if(NOT ${LIBRAW_FOUND})
    message(FATAL_ERROR "** Unable to locate LibRaw.")
endif()

find_package(OpenCV REQUIRED) 
if(NOT ${OPENCV_FOUND})
    message(FATAL_ERROR "** Unable to locate OpenCV.")
else()
    message("** Using OpenCV ${OpenCV_Version}")
endif()

find_package(xtensor REQUIRED)
if(NOT ${XTENSOR_FOUND})
    message(FATAL_ERROR "** Unable to locate Xtensor.")
else()
    message("** Using Xtensor")
endif()

find_package(xtensor-blas REQUIRED)

add_executable(main main.cpp)
target_compile_definitions(main PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(main PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${RC_INCLUDE_DIR}) 

if(CMAKE_BUILD_TYPE MATCHES Debug)
if(${LIBRAW_FOUND})
    # message("** Using LibRaw ${LibRaw_LIBRARY_DEBUG}")
    message("** Using LibRaw ${LibRaw_LIBRARIES}")
endif()
# target_link_libraries(main PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARY_DEBUG} ${OpenCV_LIBS})
target_link_libraries(main PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARIES} ${OpenCV_LIBS})  
else()
if(${LIBRAW_FOUND})
    message("** Using LibRaw ${LibRaw_LIBRARY_RELEASE}")
endif()
target_link_libraries(main PRIVATE xtensor xtensor-blas ${LibRaw_LIBRARY_RELEASE} ${OpenCV_LIBS}) 
endif()
#add_executable(sample sample.cpp)
#target_compile_definitions(sample PRIVATE ${LibRaw_DEFINITIONS})
#target_include_directories(sample PRIVATE ${LibRaw_INCLUDE_DIR})
#target_link_libraries(sample PRIVATE ${LibRaw_LIBRARIES})



set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    target_compile_options(main PRIVATE /EHsc /MP /bigobj "/std:c++17" )
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

