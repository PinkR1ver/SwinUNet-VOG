cmake_minimum_required(VERSION 3.10)
project(SimpleRecorder VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置MFC支持（如果使用MSVC）
if(MSVC)
    set(CMAKE_MFC_FLAG 2) # 使用MFC动态库
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR})

# 源文件
set(SOURCES
    SimpleRecorder.cpp
    cap_card.h
)

# 可执行文件
add_executable(SimpleRecorder ${SOURCES})

# 链接CapLib库
target_link_libraries(SimpleRecorder PRIVATE
    CapLib.lib
)

# 设置输出目录
set_target_properties(SimpleRecorder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制DLL文件（如果需要）
if(EXISTS "${CMAKE_SOURCE_DIR}/../MFC-demo/MFCApplication4/MFCApplication2/CapLib.dll")
    configure_file(
        "${CMAKE_SOURCE_DIR}/../MFC-demo/MFCApplication4/MFCApplication2/CapLib.dll"
        "${CMAKE_BINARY_DIR}/bin/CapLib.dll"
        COPYONLY
    )
endif()
