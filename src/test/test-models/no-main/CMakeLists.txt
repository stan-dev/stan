file(GLOB_RECURSE MODELS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     *.stan)

set(NOMAIN_SRC "")
foreach(stanfile ${MODELS})
  stan_model(srcfile ${stanfile} NOMAIN NOBUILD)
  list(APPEND NOMAIN_SRC "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
endforeach()
add_custom_target(Test_NoMain_Models SOURCES ${NOMAIN_SRC})

