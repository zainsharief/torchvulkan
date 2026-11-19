set(SHADER_GENERATOR "${CMAKE_SOURCE_DIR}/scripts/build_shaders.py")
set(SHADER_PACKER "${CMAKE_SOURCE_DIR}/scripts/pack_shaders.py")
set(SHADER_SPLICER "${CMAKE_SOURCE_DIR}/scripts/splice_coop_dims.py")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SHADER_GENERATOR}"
    "${CMAKE_SOURCE_DIR}/scripts/config.py")

set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR} ${TORCHVULKAN_GENERATED_DIR})

message(STATUS "Running shader generator: ${SHADER_GENERATOR}")
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SHADER_GENERATOR}
        --out ${SHADER_OUT_DIR}
        --kernels-header ${TORCHVULKAN_GENERATED_DIR}/shader_kernels.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE SHADER_GENERATOR_RESULT
)

if (NOT SHADER_GENERATOR_RESULT EQUAL 0)
    message(FATAL_ERROR "Shader generator failed (exit ${SHADER_GENERATOR_RESULT})")
endif()

file(GLOB_RECURSE SHADER_MODULES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/shaders/core/*.slang"
     "${CMAKE_SOURCE_DIR}/src/shaders/helpers/*.slang"
     "${CMAKE_SOURCE_DIR}/src/shaders/ops/*.slang")

set(SPV_FILES "")

function(torchvulkan_add_shader NAME SOURCE ENTRY)
    cmake_parse_arguments(SHADER "" "" "COOP_DIMS;SPECIALIZE" ${ARGN})
    set(SPECIALIZE_ARGS "")
    foreach(ARG ${SHADER_SPECIALIZE})
        list(APPEND SPECIALIZE_ARGS -specialize ${ARG})
    endforeach()

    set(SPV_OUT "${SHADER_OUT_DIR}/${NAME}.spv")

    # cooperative-matrix kernels compile to a raw .spv that scripts/splice_coop_dims.py finishes
    set(SLANG_OUT ${SPV_OUT})
    set(SPLICE "")
    if (SHADER_COOP_DIMS)
        list(GET SHADER_COOP_DIMS 0 COOP_SPEC_ID)
        list(GET SHADER_COOP_DIMS 1 COOP_PLACEHOLDER)
        set(SLANG_OUT "${SHADER_OUT_DIR}/${NAME}.raw.spv")
        set(SPLICE COMMAND ${Python3_EXECUTABLE} ${SHADER_SPLICER} ${SLANG_OUT} ${SPV_OUT} --spec-id ${COOP_SPEC_ID} --placeholder ${COOP_PLACEHOLDER})
    endif()

    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${SLANGC_EXE} ${SOURCE} -entry ${ENTRY} ${SPECIALIZE_ARGS} -target spirv -capability SPV_KHR_cooperative_matrix+SPV_EXT_physical_storage_buffer -I ${CMAKE_SOURCE_DIR}/src/shaders -g0 -Wno-39001 -o ${SLANG_OUT}
        ${SPLICE}
        DEPENDS ${SOURCE} ${SHADER_MODULES} ${SHADER_SPLICER}
        COMMENT "Compiling shader ${NAME}"
    )

    set(SPV_FILES ${SPV_FILES} ${SPV_OUT} PARENT_SCOPE)
endfunction()

include("${SHADER_OUT_DIR}/variants.cmake")

add_custom_command(
    OUTPUT ${TORCHVULKAN_SHADER_REGISTRY_CPP} ${TORCHVULKAN_GENERATED_DIR}/shader_registry.h
    COMMAND ${Python3_EXECUTABLE} ${SHADER_PACKER}
        --build ${SHADER_OUT_DIR}
        --variants ${SHADER_OUT_DIR}/variants.json
        --out ${TORCHVULKAN_SHADER_REGISTRY_CPP}
        --header ${TORCHVULKAN_GENERATED_DIR}/shader_registry.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${SPV_FILES} ${SHADER_PACKER} "${SHADER_OUT_DIR}/variants.json"
    COMMENT "Packing shaders into ${TORCHVULKAN_SHADER_REGISTRY_CPP}"
)

add_custom_target(compile_shaders ALL DEPENDS ${TORCHVULKAN_SHADER_REGISTRY_CPP})
