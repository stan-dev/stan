file(GLOB_RECURSE MODELS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     *.stan)

set(SYNTAXONLY_SRCS "")
foreach(stanfile ${MODELS})
  configure_file(${stanfile} ${stanfile})
  stan_model(srcfile ${stanfile} NOBUILD)
  string(REPLACE ".cpp" "${empty_string}" executable ${srcfile})
  string(REPLACE "/"  "_" executable ${executable})
  string(REPLACE "\\"  "_" executable ${executable})
  set(executable stan_test_testmodels_syntaxonly_${executable})

  list(APPEND SYNTAXONLY_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
  if(HAVE_SYNTAXONLY_FLAG)
    add_test(NAME ${executable} 
             COMMAND ${CMAKE_CXX_COMPILER_WITH_PATH} ${srcfile} ${CMAKE_CXX_FLAGS} -O0 -fsyntax-only) 
  endif()
endforeach()
add_custom_target(Test_SyntaxOnly_Models SOURCES ${SYNTAXONLY_SRCS})

