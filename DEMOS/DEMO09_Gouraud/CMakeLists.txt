cmake_minimum_required(VERSION 3.5)

project(demo9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

add_executable(demo9 main.cpp)

# Include directories
target_include_directories(demo9 PRIVATE
    ${OPENGL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${glm_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(demo9
    OpenGL::GL
    glfw
    GLEW::GLEW
)

# On Windows, we need to link against additional libraries
if(WIN32)
    target_link_libraries(demo9
        opengl32
        gdi32
    )
endif()