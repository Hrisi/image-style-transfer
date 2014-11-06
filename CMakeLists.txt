cmake_minimum_required (VERSION 2.8)
project ("Image Style Transfer")

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable (style_transfer
  ${SRC_FILES})

target_link_libraries (style_transfer ${OpenCV_LIBS})
