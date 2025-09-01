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

set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_VISIBILITY_PRESET hidden
)

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)
target_include_directories(${PLUGIN_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

add_library(${PLUGIN_NAME}_c_api SHARED
  "face_detection_tflite_plugin_c_api.cpp"
)
target_include_directories(${PLUGIN_NAME}_c_api PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME}_c_api PRIVATE ${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME}_c_api PROPERTIES OUTPUT_NAME "${PLUGIN_NAME}")

set(FACE_DETECTION_TFLITE_DLL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite/libtensorflowlite_c-win.dll")
set(face_detection_tflite_bundled_libraries "${FACE_DETECTION_TFLITE_DLL}" PARENT_SCOPE)
