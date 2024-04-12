cmake_minimum_required(VERSION 3.0)
project(RANSAC_Line_Fitting)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(test_ransac_line test_ransac_line.cpp)

# 链接 math 库
target_link_libraries(test_ransac_line m)
