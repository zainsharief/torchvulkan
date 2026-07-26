file(GLOB SHADER_SCRIPTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/scripts/build_shaders.py")

set(SHADER_OUT_DIR "${CMAKE_SOURCE_DIR}/build/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

foreach (SCRIPT ${SHADER_SCRIPTS})
    message(STATUS "Running shader generator: ${SCRIPT}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endforeach()

file(GLOB GENERATED_SLANG_FILES CONFIGURE_DEPENDS "${SHADER_OUT_DIR}/*.slang")
file(GLOB BASE_SLANG_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/shaders/*.slang")

set(SPV_FILES "")

foreach(SLANG_FILE ${GENERATED_SLANG_FILES})
    get_filename_component(SLANG_NAME ${SLANG_FILE} NAME_WE)
    
    set(SPV_OUT "${SHADER_OUT_DIR}/${SLANG_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${SLANGC_EXE} ${SLANG_FILE} -target spirv -capability SPV_KHR_cooperative_matrix -fvk-use-entrypoint-name -I ${CMAKE_SOURCE_DIR}/src/shaders -g0 -Wno-39001 -o ${SPV_OUT}
        DEPENDS ${SLANG_FILE} ${BASE_SLANG_FILES}
        COMMENT "Compiling generated shader: ${SLANG_NAME}.slang"
    )

    list(APPEND SPV_FILES ${SPV_OUT})
endforeach()

set(SHADER_REGISTRY_H "${CMAKE_SOURCE_DIR}/src/shaders/shader_registry.h")

add_custom_command(
    OUTPUT ${SHADER_REGISTRY_H}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/pack_shaders.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${SPV_FILES} ${CMAKE_SOURCE_DIR}/scripts/pack_shaders.py
    COMMENT "Packing shaders into ${SHADER_REGISTRY_H}"
)

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_REGISTRY_H})