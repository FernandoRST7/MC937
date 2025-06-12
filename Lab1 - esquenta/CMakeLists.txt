cmake_minimum_required(VERSION 3.5)

project(Lab1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR D:/OpenCV/opencv/build)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} )


add_executable(Lab1 main.cpp)

target_link_libraries(Lab1 ${OpenCV_LIBS} )