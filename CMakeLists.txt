cmake_minimum_required(VERSION 3.9)
project(eye_tracking)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
add_executable(eye_tracking main.cpp)
target_link_libraries( eye_tracking ${OpenCV_LIBS} )