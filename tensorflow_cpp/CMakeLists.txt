cmake_minimum_required(VERSION 3.10)
project(example)

include_directories(include/Eigen)
include_directories(include/fdeep)
include_directories(include/fplus)
include_directories(include/nlohmann)
include_directories(include/tensorflow)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

set(CMAKE_CXX_STANDARD 17)

add_executable(example main.cpp)
target_include_directories(example PRIVATE /path/to/cppflow/include/ $ENV{HOME}/libtensorflow2/include)
target_link_libraries (example "${TENSORFLOW_LIB}")
