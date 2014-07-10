file(GLOB_RECURSE MODELS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     *.stan)

set(SYNTAXONLY_SRCS "")
foreach(stanfile ${MODELS})
  stan_model(srcfile ${stanfile} NOBUILD)
  list(APPEND SYNTAXONLY_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
  if(HAVE_SYNTAXONLY_FLAG)
    add_test(test_models_syntaxonly
             ${CMAKE_CXX_COMPILER_WITH_PATH} ${CMAKE_CXX_FLAGS} -O0 -fsyntax-only ${srcfile})
  endif()
endforeach()
add_custom_target(Test_SyntaxOnly_Models SOURCES ${SYNTAXONLY_SRCS})

