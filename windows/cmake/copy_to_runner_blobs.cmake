# Inputs:
#  - CANDIDATES: semicolon-separated list of candidate runner dirs (no trailing /blobs)
#  - SRC_DLL: path to libtensorflowlite_c-win.dll in your repo

if(NOT DEFINED CANDIDATES OR "${CANDIDATES}" STREQUAL "")
  message(FATAL_ERROR "No CANDIDATES provided to copy_to_runner_blobs.cmake")
endif()

# Prefer the first candidate; if it exists, great; if not, create it.
list(GET CANDIDATES 0 _picked)

set(_dest "${_picked}/blobs")
execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dest}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SRC_DLL}" "${_dest}/libtensorflowlite_c-win.dll")
message(STATUS "Mirrored TFLite into: ${_dest}/libtensorflowlite_c-win.dll")
