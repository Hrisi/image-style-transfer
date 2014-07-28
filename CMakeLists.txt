cmake_minimum_required (VERSION 2.8)
project ("Image Style Transfer")

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable (sample_program
    "${CMAKE_CURRENT_SOURCE_DIR}/Sample/sample_code.cpp")

target_link_libraries (sample_program ${OpenCV_LIBS})
