set(SHADER_GENERATOR "${CMAKE_SOURCE_DIR}/scripts/build_shaders.py")
set(SHADER_PACKER "${CMAKE_SOURCE_DIR}/scripts/pack_shaders.py")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SHADER_GENERATOR}"
    "${CMAKE_SOURCE_DIR}/scripts/config.py")

set(SHADER_OUT_DIR "${CMAKE_SOURCE_DIR}/build/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

message(STATUS "Running shader generator: ${SHADER_GENERATOR}")
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SHADER_GENERATOR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE SHADER_GENERATOR_RESULT
)

if (NOT SHADER_GENERATOR_RESULT EQUAL 0)
    message(FATAL_ERROR "Shader generator failed (exit ${SHADER_GENERATOR_RESULT})")
endif()

file(GLOB_RECURSE SHADER_MODULES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/shaders/core/*.slang"
     "${CMAKE_SOURCE_DIR}/src/shaders/ops/*.slang")

set(SPV_FILES "")

function(torchvulkan_add_shader NAME SOURCE ENTRY)
    set(SPECIALIZE_ARGS "")
    foreach(ARG ${ARGN})
        list(APPEND SPECIALIZE_ARGS -specialize ${ARG})
    endforeach()

    set(SPV_OUT "${SHADER_OUT_DIR}/${NAME}.spv")

    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${SLANGC_EXE} ${SOURCE} -entry ${ENTRY} ${SPECIALIZE_ARGS} -target spirv -capability SPV_KHR_cooperative_matrix+SPV_EXT_physical_storage_buffer -I ${CMAKE_SOURCE_DIR}/src/shaders -g0 -Wno-39001 -o ${SPV_OUT}
        DEPENDS ${SOURCE} ${SHADER_MODULES}
        COMMENT "Compiling shader ${NAME}"
    )

    set(SPV_FILES ${SPV_FILES} ${SPV_OUT} PARENT_SCOPE)
endfunction()

include("${SHADER_OUT_DIR}/variants.cmake")

set(SHADER_REGISTRY_H "${CMAKE_SOURCE_DIR}/src/shaders/shader_registry.h")

add_custom_command(
    OUTPUT ${SHADER_REGISTRY_H}
    COMMAND ${Python3_EXECUTABLE} ${SHADER_PACKER}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${SPV_FILES} ${SHADER_PACKER} "${SHADER_OUT_DIR}/variants.json"
    COMMENT "Packing shaders into ${SHADER_REGISTRY_H}"
)

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_REGISTRY_H})
