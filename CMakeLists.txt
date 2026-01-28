cmake_minimum_required(VERSION 3.18)
project(sample_onnx_ep VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Find ONNX Runtime
# ============================================================================

# Option to specify custom ONNX Runtime path
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
else()
    # Try to find in standard locations
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_c_api.h
        PATHS
            /usr/local/include/onnxruntime
            /usr/include/onnxruntime
            /opt/onnxruntime/include
        PATH_SUFFIXES
            onnxruntime/core/session
    )

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            /usr/local/lib
            /usr/lib
            /opt/onnxruntime/lib
    )

    if(ONNXRUNTIME_LIBRARY)
        get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
    endif()
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR
        "ONNX Runtime headers not found. Please set ONNXRUNTIME_ROOT to the "
        "ONNX Runtime installation directory, or install onnxruntime-dev.")
endif()

message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib dir: ${ONNXRUNTIME_LIB_DIR}")

# ============================================================================
# Build the Plugin EP Shared Library
# ============================================================================

add_library(sample_ep SHARED
    src/sample_ep.cpp
)

target_include_directories(sample_ep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Set visibility to hidden by default, only export what we need
if(NOT MSVC)
    target_compile_options(sample_ep PRIVATE
        -fvisibility=hidden
        -Wall
        -Wextra
    )
endif()

# Platform-specific settings
if(WIN32)
    set_target_properties(sample_ep PROPERTIES
        PREFIX ""
        OUTPUT_NAME "sample_ep"
        SUFFIX ".dll"
    )
elseif(APPLE)
    set_target_properties(sample_ep PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "sample_ep"
        SUFFIX ".dylib"
    )
else()
    set_target_properties(sample_ep PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "sample_ep"
        SUFFIX ".so"
    )
endif()

# ============================================================================
# Build a Test Application (optional)
# ============================================================================

option(BUILD_TEST_APP "Build the test application" ON)

if(BUILD_TEST_APP AND ONNXRUNTIME_LIB_DIR)
    add_executable(test_sample_ep
        test/test_sample_ep.cpp
    )

    target_include_directories(test_sample_ep PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
    )

    target_link_directories(test_sample_ep PRIVATE
        ${ONNXRUNTIME_LIB_DIR}
    )

    target_link_libraries(test_sample_ep PRIVATE
        onnxruntime
    )

    # Copy the plugin library next to the test executable for easy testing
    add_custom_command(TARGET test_sample_ep POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:sample_ep>
            $<TARGET_FILE_DIR:test_sample_ep>
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS sample_ep
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
