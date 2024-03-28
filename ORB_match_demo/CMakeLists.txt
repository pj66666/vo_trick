cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
"/usr/include/eigen3"
)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        )
add_subdirectory(src)
add_subdirectory(test)



