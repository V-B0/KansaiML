cmake_minimum_required(VERSION 3.18)
if(APPLE)
  project(kansai LANGUAGES CXX OBJCXX)
else()
  project(kansai LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NB_DIR_RESULT
)
if(NOT NB_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "nanobind not found for ${Python_EXECUTABLE}. Run: python3 -m pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(backend/cpu)
if(APPLE)
  add_subdirectory(backend/metal)
endif()
add_subdirectory(core)
add_subdirectory(python)
