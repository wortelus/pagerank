cmake_minimum_required(VERSION 3.26)
project(pagerank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Nastav buď Release nebo vypiš současný build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
else ()
    message("Build type: ${CMAKE_BUILD_TYPE}")
endif ()

message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)

add_executable(pagerank main.cpp dd_pagerank.cpp dd_pagerank.h edge.h consts.h)

if (OpenMP_CXX_FOUND)
    target_link_libraries(pagerank PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pagerank PRIVATE USE_OPENMP)  # Definice pro překladač
endif ()

# Optimalizace
if (MSVC)
    message(STATUS "Using MSVC")
    target_compile_options(pagerank PRIVATE
            $<$<CONFIG:Release>:/O2 /Ob2 /Oi /Ot /GL> # Optimalizace
            $<$<CONFIG:Debug>:/Od /Zi>                # Debug info, bez optimalizace
    )
    target_link_options(pagerank PRIVATE
            $<$<CONFIG:Release>:/LTCG>                # Link-time code generation
    )

    # OpenMP flagy pro MSVC
    if (OpenMP_CXX_FOUND)
        target_compile_options(pagerank PRIVATE /openmp:llvm) # LLVM backend
    endif ()

elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Using GCC or Clang")
    target_compile_options(pagerank PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
    )
    # Odkomentuj pro LTO
    # target_link_options(pagerank PRIVATE $<$<CONFIG:Release>:-flto>)  # Enable LTO
    if (OpenMP_CXX_FOUND)
        target_compile_options(pagerank PRIVATE -fopenmp)
        target_link_libraries(pagerank PUBLIC -fopenmp)
    endif ()
endif ()
