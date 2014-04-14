# no-main models
file(GLOB_RECURSE NOMAIN_MODELS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     no-main/*.stan)

set(NOMAIN_SRCS "")
foreach(stanfile ${NOMAIN_MODELS})
  stan_model(srcfile ${stanfile} NOMAIN NOBUILD)
  list(APPEND NOMAIN_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
endforeach()
set(NOMAIN_MODEL_SRCS ${NOMAIN_SRCS} PARENT_SCOPE)
#add_custom_target(Test_NoMain_Models SOURCES ${NOMAIN_SRCS})

# syntax-only models
file(GLOB_RECURSE SYNTAXONLY_MODELS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     syntax-only/*.stan)

set(SYNTAXONLY_SRCS "")
foreach(stanfile ${SYNTAXONLY_MODELS})
  stan_model(srcfile ${stanfile} NOBUILD)
  list(APPEND SYNTAXONLY_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${srcfile}")
  if(HAVE_SYNTAXONLY_FLAG)
    add_test(test_models_syntaxonly
             ${CMAKE_CXX_COMPILER_WITH_PATH} ${CMAKE_CXX_FLAGS} -O0 -fsyntax-only ${srcfile})
  endif()
endforeach()
set(SYNTAXONLY_MODEL_SRCS ${SYNTAXONLY_SRCS} PARENT_SCOPE)
# This next line causes the stan files to be parsed every time, we probably
# shouldn't use it...
add_custom_target(Test_SyntaxOnly_Models SOURCES ${SYNTAXONLY_SRCS})

