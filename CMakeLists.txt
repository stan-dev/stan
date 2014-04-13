cmake_minimum_required (VERSION 2.8)

project(Stan)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
include(StanModels)

option(USE_SYSTEM_EIGEN
       "Whether to use the system version of Eigen or the included version"
       OFF)
option(USE_SYSTEM_BOOST
       "Whether to use the system version of Boost or the included version"
       OFF)
option(USE_SYSTEM_GTEST
       "Whether to use the system version of GTest or the included version"
       OFF)

# Find Eigen
if(USE_SYSTEM_EIGEN)
  find_package( Eigen3 REQUIRED )
  include_directories("${EIGEN3_INCLUDE_DIR}")
else()
  include_directories("${PROJECT_SOURCE_DIR}/lib/eigen_3.2.0")
endif()

# Find Boost
if(USE_SYSTEM_BOOST)
  find_package( Boost 1.54 REQUIRED )
  include_directories("${Boost_INCLUDE_DIR}")
else()
  include_directories("${PROJECT_SOURCE_DIR}/lib/boost_1.54.0")
endif()

# Setup GTest library
if(USE_SYSTEM_GTEST)
  find_package( GTest REQUIRED )
  set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
else()
  add_subdirectory(lib/gtest_1.7.0)
endif()

# Setup compiler flags
check_cxx_compiler_flag("-pipe" HAVE_PIPE_FLAG)
if(HAVE_PIPE_FLAG)
  add_compile_options( -pipe )
endif(HAVE_PIPE_FLAG)

check_cxx_compiler_flag("-Wall" HAVE_WALL_FLAG)
if(HAVE_WALL_FLAG)
  add_compile_options( -Wall )
endif(HAVE_WALL_FLAG)

# Setup boost defines
add_definitions(-DBOOST_RESULT_OF_USE_TR1
                -DBOOST_NO_DECLTYPE
                -DBOOST_DISABLE_ASSERTS)
# Top level include
include_directories("${PROJECT_SOURCE_DIR}/src")

# add src/stan/gm directory to build libstanc.a
add_subdirectory(src/stan/gm)

# libstan.a
set(LIBSTAN_SRCS ${PROJECT_SOURCE_DIR}/src/stan/agrad/rev/var_stack.cpp)
add_library(stan STATIC ${LIBSTAN_SRCS})

# stanc
set(STANC_SRCS ${PROJECT_SOURCE_DIR}/src/stan/command/stanc.cpp)
add_executable(stanc-bin ${STANC_SRCS})
set_target_properties(stanc-bin PROPERTIES OUTPUT_NAME stanc)
target_link_libraries(stanc-bin stanc)
set(STANC_BIN ${CMAKE_CURRENT_BINARY_DIR}/stanc)

# print 
set(PRINT_SRCS ${PROJECT_SOURCE_DIR}/src/stan/command/print.cpp)
add_executable(print ${PRINT_SRCS})
target_link_libraries(print stan)
set(PRINT_BIN ${CMAKE_CURRENT_BINARY_DIR}/print)

# unit tests
include(CTest)
add_subdirectory(src/test)

