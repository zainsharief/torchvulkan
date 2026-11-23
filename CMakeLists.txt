cmake_minimum_required(VERSION 3.18)
project(torchvulkan LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_program(SLANGC_EXE slangc HINTS "$ENV{SLANG_DIR}" REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCHVULKAN_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(TORCHVULKAN_SHADER_REGISTRY_CPP "${TORCHVULKAN_GENERATED_DIR}/shader_registry.cpp")
add_subdirectory(src/shaders)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
set_source_files_properties(${TORCHVULKAN_SHADER_REGISTRY_CPP} PROPERTIES GENERATED TRUE)

python3_add_library(torchvulkan_lib MODULE ${SOURCES} ${TORCHVULKAN_SHADER_REGISTRY_CPP} WITH_SOABI)
add_dependencies(torchvulkan_lib compile_shaders)

set_target_properties(torchvulkan_lib PROPERTIES OUTPUT_NAME "_C")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
target_link_libraries(torchvulkan_lib PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Module ${CMAKE_DL_LIBS})
target_include_directories(torchvulkan_lib PRIVATE src/ ${CMAKE_BINARY_DIR} extern/vma extern/volk ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(torchvulkan_lib PRIVATE VK_NO_PROTOTYPES TORCH_EXTENSION_NAME=_C)

target_compile_features(torchvulkan_lib PRIVATE cxx_std_20)

install(TARGETS torchvulkan_lib DESTINATION torchvulkan)

file(WRITE "${TORCHVULKAN_GENERATED_DIR}/_build_info.py" "torch_version = \"${Torch_VERSION}\"\n")
install(FILES "${TORCHVULKAN_GENERATED_DIR}/_build_info.py" DESTINATION torchvulkan)

if(APPLE)
    # a path an older configure's find_library cached is no choice; -DTORCHVULKAN_MOLTENVK=<path> still wins
    get_property(TORCHVULKAN_MOLTENVK_HELP CACHE TORCHVULKAN_MOLTENVK PROPERTY HELPSTRING)
    if(TORCHVULKAN_MOLTENVK_HELP STREQUAL "Path to a library.")
        unset(TORCHVULKAN_MOLTENVK CACHE)
    endif()

    # the pinned MoltenVK release with cooperative matrices, the one CI bundles
    set(TORCHVULKAN_MOLTENVK_FETCH "${CMAKE_SOURCE_DIR}/.github/scripts/fetch-moltenvk-macos.sh")
    if(NOT TORCHVULKAN_MOLTENVK AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TORCHVULKAN_MOLTENVK_FETCH}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "MOLTENVK_OUT_DIR=${CMAKE_BINARY_DIR}/moltenvk" bash "${TORCHVULKAN_MOLTENVK_FETCH}"
            RESULT_VARIABLE TORCHVULKAN_MOLTENVK_RESULT
        )
        if(NOT TORCHVULKAN_MOLTENVK_RESULT EQUAL 0)
            message(FATAL_ERROR "Fetching MoltenVK failed; pass -DTORCHVULKAN_MOLTENVK=<path to libMoltenVK.dylib> to bundle another.")
        endif()
        set(TORCHVULKAN_MOLTENVK "${CMAKE_BINARY_DIR}/moltenvk/libMoltenVK.dylib")
    else()
        find_library(TORCHVULKAN_MOLTENVK NAMES libMoltenVK.dylib)
    endif()
    if(TORCHVULKAN_MOLTENVK)
        get_filename_component(TORCHVULKAN_MOLTENVK_REAL "${TORCHVULKAN_MOLTENVK}" REALPATH)
        message(STATUS "Bundling MoltenVK: ${TORCHVULKAN_MOLTENVK_REAL}")
        install(FILES "${TORCHVULKAN_MOLTENVK_REAL}" DESTINATION torchvulkan RENAME libMoltenVK.dylib)
    else()
        message(WARNING "libMoltenVK.dylib not found; torchvulkan will need a system Vulkan loader at runtime.")
    endif()
endif()