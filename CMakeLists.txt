cmake_minimum_required(VERSION 3.26)
project(pagerank)

set(CMAKE_CXX_STANDARD 20)

message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("Found OpenMP!")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Not found OpenMP!")
ENDIF()

add_executable(pagerank main.cpp
        dd_pagerank.cpp
        dd_pagerank.h
        edge.h
        consts.h)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pagerank PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(pagerank PRIVATE /openmp)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pagerank PRIVATE -fopenmp)
endif()
