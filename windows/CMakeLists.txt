cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "face_detection_tflite")
project(${PROJECT_NAME} LANGUAGES CXX)

cmake_policy(VERSION 3.14...3.25)

set(PLUGIN_NAME "face_detection_tflite_plugin")

# Sources
list(APPEND PLUGIN_SOURCES
  "face_detection_tflite_plugin.cpp"
  "face_detection_tflite_plugin.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/face_detection_tflite/face_detection_tflite_plugin_c_api.h"
  "face_detection_tflite_plugin_c_api.cpp"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Link only to Flutter bits; TFLite C is loaded by Dart via FFI, not linked here.
target_link_libraries(${PLUGIN_NAME} PRIVATE
  flutter
  flutter_wrapper_plugin
)

# --- Option A: bundle TFLite C into a blobs/ dir beside the runner EXE (Windows) ---

if (WIN32)
  # Expect the prebuilt DLL to live in the plugin at: <plugin>/blobs/libtensorflowlite_c-win.dll
  set(TFLITE_DLL_WIN "${CMAKE_CURRENT_SOURCE_DIR}/blobs/libtensorflowlite_c-win.dll")

  if (NOT EXISTS "${TFLITE_DLL_WIN}")
    message(FATAL_ERROR "Missing ${TFLITE_DLL_WIN}. Place libtensorflowlite_c-win.dll in the plugin's blobs/ folder.")
  endif()

  message(STATUS "Will copy ${TFLITE_DLL_WIN} into the app bundle blobs/ directory.")

  # Copy to: .../build/windows/x64/runner/<CONFIG>/blobs/libtensorflowlite_c-win.dll
  add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_FILE_DIR:${PLUGIN_NAME}>/../runner/$<CONFIG>/blobs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${TFLITE_DLL_WIN}"
      "$<TARGET_FILE_DIR:${PLUGIN_NAME}>/../runner/$<CONFIG>/blobs/libtensorflowlite_c-win.dll"
  )
endif()

# === Tests ===
if (${include_${PROJECT_NAME}_tests})
  set(TEST_RUNNER "${PROJECT_NAME}_test")
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.11.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(${TEST_RUNNER}
    test/face_detection_tflite_plugin_test.cpp
    ${PLUGIN_SOURCES}
  )
  apply_standard_settings(${TEST_RUNNER})
  target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${TEST_RUNNER} PRIVATE flutter_wrapper_plugin gtest_main gmock)

  add_custom_command(TARGET ${TEST_RUNNER} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${FLUTTER_LIBRARY}" $<TARGET_FILE_DIR:${TEST_RUNNER}>
    # Also copy the blobs DLL for tests, matching the app layout
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_FILE_DIR:${TEST_RUNNER}>/blobs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${TFLITE_DLL_WIN}" "$<TARGET_FILE_DIR:${TEST_RUNNER}>/blobs/libtensorflowlite_c-win.dll"
  )

  include(GoogleTest)
  gtest_discover_tests(${TEST_RUNNER})
endif()
