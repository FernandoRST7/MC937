cmake_minimum_required(VERSION 3.5)

project(Lab2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

# Add glm (assuming it's in a standard location or installed system-wide)
find_package(glm REQUIRED)

add_executable(Lab2 main.cpp)

# Include directories
target_include_directories(Lab2 PRIVATE
    ${OPENGL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${glm_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(Lab2
    OpenGL::GL
    glfw
    GLEW::GLEW
)   