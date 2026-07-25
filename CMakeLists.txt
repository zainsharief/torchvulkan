cmake_minimum_required(VERSION 3.18)
project(torchvulkan LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_program(SLANGC_EXE slangc REQUIRED)
find_program(SPIRV_CROSS_EXE spirv-cross REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/shaders)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

python3_add_library(torchvulkan_lib MODULE ${SOURCES} WITH_SOABI)
add_dependencies(torchvulkan_lib compile_shaders)

set_target_properties(torchvulkan_lib PROPERTIES OUTPUT_NAME "_C" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torchvulkan")
target_link_libraries(torchvulkan_lib PRIVATE ${TORCH_LIBRARIES} Python3::Module ${CMAKE_DL_LIBS})
target_include_directories(torchvulkan_lib PRIVATE src/ extern/vma extern/volk ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(torchvulkan_lib PRIVATE VK_NO_PROTOTYPES TORCH_EXTENSION_NAME=_C)

target_compile_features(torchvulkan_lib PRIVATE cxx_std_20)

install(TARGETS torchvulkan_lib DESTINATION torchvulkan)