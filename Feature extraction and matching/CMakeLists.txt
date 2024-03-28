cmake_minimum_required(VERSION 3.0)
project(FeatureMatching)

# 添加 OpenCV 包
find_package(OpenCV 3 REQUIRED)

# 设置可执行文件生成路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(${OpenCV_INCLUDE_DIRS})


# 添加可执行文件
add_executable(BFMatch BFMatch.cpp)
target_link_libraries(BFMatch ${OpenCV_LIBS})

add_executable(DistanceMatch DistanceMatch.cpp)
target_link_libraries(DistanceMatch ${OpenCV_LIBS})

add_executable(knnMatch knnMatch.cpp)
target_link_libraries(knnMatch ${OpenCV_LIBS})

add_executable(RansacMatch RansacMatch.cpp)
target_link_libraries(RansacMatch ${OpenCV_LIBS})
