# Stan interfaces should include this after the stan subdirectory has been
# included to setup the correct compile flags

# Setup compiler flags
check_cxx_compiler_flag("-pipe" HAVE_PIPE_FLAG)
if(HAVE_PIPE_FLAG)
  add_compile_options( -pipe )
endif(HAVE_PIPE_FLAG)

# FIXME: If any of these warning flags will break compilers, we can check for
# them like is being done with -pipe
add_compile_options( -Wall -Wno-unused-function -Wno-tautological-compare
                     -Wno-c++11-long-long )

# This is used for the syntax-only test compiles
#check_cxx_compiler_flag("-fsyntax-only" HAVE_SYNTAXONLY_FLAG)
set(HAVE_SYNTAXONLY_FLAG ON)

# Setup boost defines
add_definitions(-DBOOST_RESULT_OF_USE_TR1
                -DBOOST_NO_DECLTYPE
                -DBOOST_DISABLE_ASSERTS)

# Top level include
include_directories("${Stan_SOURCE_DIR}/src" "${Stan_BINARY_DIR}/src")


