cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "face_detection_tflite")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "face_detection_tflite_plugin")

add_library(${PLUGIN_NAME} SHARED
  "face_detection_tflite_plugin.cpp"
  "face_detection_tflite_plugin.h"
  "face_detection_tflite_plugin_c_api.cpp"
  "include/face_detection_tflite/face_detection_tflite_plugin_c_api.h"
)

apply_standard_settings(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)
target_include_directories(${PLUGIN_NAME}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

set(FACE_DETECTION_TFLITE_DLL "${CMAKE_CURRENT_SOURCE_DIR}/../assets/bin/libtensorflowlite_c-win.dll")
set(${PROJECT_NAME}_bundled_libraries "${FACE_DETECTION_TFLITE_DLL}" PARENT_SCOPE)