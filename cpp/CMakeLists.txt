cmake_minimum_required(VERSION 3.16)
project(da360_onnx_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenCV (Homebrew) ──────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")

# ── ONNX Runtime (Homebrew) ────────────────────────────────────────────────────
# Default matches Apple Silicon Homebrew prefix. If headers/libs live under the keg only:
#   cmake -S . -B build -DORT_ROOT=$(brew --prefix onnxruntime)
set(ORT_ROOT /opt/homebrew CACHE PATH "ONNX Runtime install prefix")

if(APPLE)
    set(_ort_lib "${ORT_ROOT}/lib/libonnxruntime.dylib")
elseif(UNIX)
    set(_ort_lib "${ORT_ROOT}/lib/libonnxruntime.so")
else()
    set(_ort_lib "${ORT_ROOT}/lib/onnxruntime.lib")
endif()

if(NOT EXISTS "${_ort_lib}")
    message(FATAL_ERROR
        "ONNX Runtime library not found at ${_ort_lib}\n"
        "Set ORT_ROOT to the install prefix (folder containing lib/ and include/onnxruntime/), e.g.\n"
        "  cmake -S . -B build -DORT_ROOT=$(brew --prefix onnxruntime)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include/onnxruntime")

# ── Executable ─────────────────────────────────────────────────────────────────
add_executable(da360_onnx_test da360_onnx_test.cpp)

target_link_libraries(da360_onnx_test PRIVATE ${OpenCV_LIBS} onnxruntime)

if(APPLE)
    set_target_properties(da360_onnx_test PROPERTIES
        BUILD_RPATH "${ORT_ROOT}/lib"
        INSTALL_RPATH "${ORT_ROOT}/lib")
endif()
