
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
project(example)


set(CMAKE_CXX_STANDARD 17)

find_package(jsoncpp CONFIG REQUIRED)
find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${JSON_INC_PATH})

add_executable(example main.cpp)
target_include_directories(example PRIVATE include/struct_mapping)
target_link_libraries(example PRIVATE "${TENSORFLOW_LIB}" jsoncpp_lib)


