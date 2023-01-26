find_package(LibRaw ${LIBRAW_MIN_VERSION} REQUIRED)
if(NOT ${LIBRAW_FOUND})
    message(FATAL_ERROR "** Unable to locate LibRaw... is it installed?")
else()
    message("** Using LibRaw")
endif()

find_package(OpenCV REQUIRED) 
# find_package(xtensor REQUIRED)

add_executable(main main.cpp)
target_compile_definitions(main PRIVATE ${LibRaw_DEFINITIONS})
target_include_directories(main PRIVATE ${LibRaw_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}) 
target_link_libraries(main PRIVATE ${LibRaw_LIBRARIES} ${OpenCV_LIBS}) 

#add_executable(sample sample.cpp)
#target_compile_definitions(sample PRIVATE ${LibRaw_DEFINITIONS})
#target_include_directories(sample PRIVATE ${LibRaw_INCLUDE_DIR})
#target_link_libraries(sample PRIVATE ${LibRaw_LIBRARIES})



set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    target_compile_options(main PRIVATE /EHsc /MP /bigobj "/std:c++17" )
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

