find_package(xtensor REQUIRED)
find_package(Boost REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
   message(FATAL_ERROR "No GTest")
else()
   message(STATUS "Found GTest:" ${GTEST_BOTH_LIBRARIES} " " ${GTEST_INCLUDE_DIRS})
endif()
find_package(LibRaw ${LIBRAW_MIN_VERSION} REQUIRED)
if(NOT ${LIBRAW_FOUND})
    message(FATAL_ERROR "** Unable to locate LibRaw.")
endif()

set(SOURCE test_raw_converter.cpp)

add_executable(rc_test main.cpp ${SOURCE})

if(MSVC)
    target_compile_options(rc_test PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(rc_test PRIVATE -march=native)
endif()
target_compile_definitions(rc_test PRIVATE ${LibRaw_DEFINITIONS})
include_directories(${TBB_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${LibRaw_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/rawconverter/include)
target_link_libraries(rc_test xtensor ${LibRaw_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES})

add_test(AllTests rc_test)
