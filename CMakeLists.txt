cmake_minimum_required(VERSION 3.20)
project(vkflame LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)
find_package(Python3 REQUIRED)

# ── SPIR-V kernel compilation ────────────────────────────────────
file(GLOB GLSL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.glsl")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/spirv")

set(SPV_FILES "")
foreach(GLSL_FILE ${GLSL_SOURCES})
    get_filename_component(KERNEL_NAME ${GLSL_FILE} NAME_WE)
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/spirv/${KERNEL_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLC}
            --target-env=vulkan1.3
            -O
            -fshader-stage=compute
            ${GLSL_FILE}
            -o ${SPV_FILE}
        DEPENDS ${GLSL_FILE}
        COMMENT "Compiling shader: ${KERNEL_NAME}.glsl"
    )
    list(APPEND SPV_FILES ${SPV_FILE})
endforeach()

set(SPIRV_EMBED_CPP "${CMAKE_CURRENT_BINARY_DIR}/spirv_embed.cpp")
add_custom_command(
    OUTPUT ${SPIRV_EMBED_CPP}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/spirv"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py"
            "${CMAKE_CURRENT_BINARY_DIR}/spirv"
            "${SPIRV_EMBED_CPP}"
    DEPENDS ${SPV_FILES}
    COMMENT "Embedding SPIR-V into spirv_embed.cpp"
)

add_custom_target(spirv_kernels DEPENDS ${SPV_FILES} ${SPIRV_EMBED_CPP})

# (Python3 found above, before custom commands)

# ── Runtime shared library ───────────────────────────────────────
file(GLOB RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp")
add_library(vkflame_rt SHARED
    ${RUNTIME_SOURCES}
    ${SPIRV_EMBED_CPP}
)
add_dependencies(vkflame_rt spirv_kernels)
target_include_directories(vkflame_rt PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${Vulkan_INCLUDE_DIRS}"
)
target_link_libraries(vkflame_rt PUBLIC ${Vulkan_LIBRARIES})
set_target_properties(vkflame_rt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ── Shim shared libraries ────────────────────────────────────────
add_library(hipblaslt SHARED "${CMAKE_CURRENT_SOURCE_DIR}/shim/hipblaslt_shim.cpp")
set_target_properties(hipblaslt PROPERTIES SOVERSION 0 WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(hipblaslt PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/shim"
    "${Vulkan_INCLUDE_DIRS}"
)
target_link_libraries(hipblaslt PRIVATE vkflame_rt)

add_library(hipblas SHARED "${CMAKE_CURRENT_SOURCE_DIR}/shim/hipblas_shim.cpp")
set_target_properties(hipblas PROPERTIES SOVERSION 2 WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(hipblas PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/shim"
    "${Vulkan_INCLUDE_DIRS}"
)
target_link_libraries(hipblas PRIVATE vkflame_rt)

add_library(amdhip64 SHARED "${CMAKE_CURRENT_SOURCE_DIR}/shim/hip_runtime_shim.cpp")
# On Windows, Ollama expects the file to be named amdhip64_6.dll (matches ROCm naming)
set_target_properties(amdhip64 PROPERTIES SOVERSION 6 OUTPUT_NAME "amdhip64_6" WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(amdhip64 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/shim"
    "${Vulkan_INCLUDE_DIRS}"
)
target_link_libraries(amdhip64 PRIVATE vkflame_rt)

# ── rocBLAS stub — ggml-hip.dll imports exactly one symbol: rocblas_initialize
add_library(rocblas SHARED "${CMAKE_CURRENT_SOURCE_DIR}/shim/rocblas_shim.cpp")
set_target_properties(rocblas PROPERTIES OUTPUT_NAME "rocblas" WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ── Test executables ─────────────────────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.cpp")
    add_executable(vkflame_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.cpp")
    target_include_directories(vkflame_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
        "${Vulkan_INCLUDE_DIRS}"
    )
    target_link_libraries(vkflame_tests PRIVATE vkflame_rt)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shim.cpp")
    add_executable(test_shim "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shim.cpp")
    target_include_directories(test_shim PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
        "${CMAKE_CURRENT_SOURCE_DIR}/shim"
        "${Vulkan_INCLUDE_DIRS}"
    )
    target_link_libraries(test_shim PRIVATE vkflame_rt)
endif()

# ── Install ──────────────────────────────────────────────────────
install(TARGETS vkflame_rt hipblaslt hipblas amdhip64 rocblas DESTINATION lib)
