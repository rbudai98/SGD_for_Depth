
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
project(example)


set(CMAKE_CXX_STANDARD 17)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)


add_executable(example main.cpp)
target_include_directories(example PRIVATE include/struct_mapping)
target_link_libraries(example PRIVATE "${TENSORFLOW_LIB}")


