cmake_minimum_required (VERSION 2.8)

project(Stan)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(USE_SYSTEM_EIGEN
       "Whether to use the system version of Eigen or the included version"
       OFF)
option(USE_SYSTEM_BOOST
       "Whether to use the system version of Boost or the included version"
       OFF)
option(USE_SYSTEM_GTEST
       "Whether to use the system version of GTest or the included version"
       OFF)
       
if(USE_SYSTEM_EIGEN)
  find_package( Eigen3 REQUIRED )
  include_directories("${EIGEN3_INCLUDE_DIR}")
else()
  include_directories("${PROJECT_SOURCE_DIR}/lib/eigen_3.2.0")
endif()

if(USE_SYSTEM_BOOST)
  find_package( Boost 1.54 REQUIRED )
  include_directories("${Boost_INCLUDE_DIR}")
else()
  include_directories("${PROJECT_SOURCE_DIR}/lib/boost_1.54.0")
endif()

if(USE_SYSTEM_GTEST)
  find_package( GTest REQUIRED )
  set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
else()
  add_subdirectory(lib/gtest_1.7.0)
  set(GTEST_INCLUDE_DIRS
      "${PROJECT_SOURCE_DIR}/lib/gtest_1.7.0"
      "${PROJECT_SOURCE_DIR}/lib/gtest_1.7.0/include")
  set(GTEST_LIBRARIES gtest)
endif()

include_directories("${PROJECT_SOURCE_DIR}/src")

function(set_target_no_optimize target)
  get_target_property(CURR_FLAGS ${target} COMPILE_FLAGS)
  set_target_properties(stanc PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -O0")
  message(STATUS "Disabling optimization for ${target}")
endfunction(set_target_no_optimize)

# libstanc.a
file(GLOB_RECURSE LIBSTANC_SRCS ${PROJECT_SOURCE_DIR}/src/stan/gm/*.cpp)
add_library(stanc STATIC ${LIBSTANC_SRCS})
set_target_no_optimize(stanc)

# libstan.a
set(LIBSTAN_SRCS ${PROJECT_SOURCE_DIR}/src/stan/agrad/rev/var_stack.cpp)
add_library(stan STATIC ${LIBSTAN_SRCS})

# unit tests
include(CTest)
add_subdirectory(src/test)

