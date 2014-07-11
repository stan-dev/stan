cmake_minimum_required (VERSION 2.8)

project(Stan C CXX)
set(Stan_VERSION_MAJOR "2")
set(Stan_VERSION_MINOR "3")
set(Stan_VERSION_PATCH "0")
set(Stan_VERSION "${Stan_VERSION_MAJOR}.${Stan_VERSION_MINOR}.${Stan_VERSION_PATCH}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
include(CTest)
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
option(BUILD_DOXYGEN_DOCS
       "Whether the doxygen-based documentation should be generated"
       ON)
option(BUILD_REFERENCE_MANUAL
       "Whether the Stan reference manual should be built"
       ON)

if(BUILD_REFERENCE_MANUAL)
  include(UseLATEX)

  add_subdirectory( src/docs/stan-reference )
endif()

if(BUILD_DOXYGEN_DOCS)
  find_package( Doxygen REQUIRED )

  add_subdirectory( src/doxygen ) 
endif()

# Find Eigen
if(USE_SYSTEM_EIGEN)
  find_package( Eigen3 REQUIRED )
  include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")
else()
  include_directories(SYSTEM "${Stan_SOURCE_DIR}/lib/eigen_3.2.0")
endif()

# Find Boost
if(USE_SYSTEM_BOOST)
  find_package( Boost 1.54 REQUIRED )
  include_directories(SYSTEM "${Boost_INCLUDE_DIR}")
else()
  include_directories(SYSTEM "${Stan_SOURCE_DIR}/lib/boost_1.54.0")
endif()

# Setup GTest library
if(USE_SYSTEM_GTEST)
  find_package( GTest REQUIRED )
  set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
else()
  if(BUILD_TESTING)
      add_subdirectory(lib/gtest_1.7.0)
  endif()
endif()

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
check_cxx_compiler_flag("-fsyntax-only" HAVE_SYNTAXONLY_FLAG)

# Setup boost defines
add_definitions(-DBOOST_RESULT_OF_USE_TR1
                -DBOOST_NO_DECLTYPE
                -DBOOST_DISABLE_ASSERTS)
# Top level include
include_directories("${Stan_SOURCE_DIR}/src" "${Stan_BINARY_DIR}/src")

# add src/stan/gm directory to build libstanc.a
add_subdirectory(src/stan/gm)

# libstan.a
set(LIBSTAN_SRCS ${PROJECT_SOURCE_DIR}/src/stan/agrad/rev/var_stack.cpp)
add_library(stan STATIC ${LIBSTAN_SRCS})

if(MINGW)
  check_cxx_compiler_flag("-flto=jobserver" HAVE_LTO)
  if(HAVE_LTO)
    add_compile_options( -flto=jobserver )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=jobserver")
    check_cxx_compiler_flag("-ffat-lto-objects" HAVE_FAT_LTO_FLAG)
    if(HAVE_FAT_LTO_FLAG)
      add_compile_options( -ffat-lto-objects )
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ffat-lto-objects")
    endif(HAVE_FAT_LTO_FLAG)
  endif(HAVE_LTO)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -lwinpthread")

  find_program(WINE_EXE wine DOC "Path the the wine executable")
  if(WINE_EXE STREQUAL WINE_EXE-NOTFOUND)
    message(error "Wine is required for building with mingw and could not be found")
  else()
    message(STATUS "Found Wine executable at: ${WINE_EXE}")
    set(RUNCMD "${WINE_EXE}")
  endif()
else()
  set(RUNCMD "")
endif()

# unit tests
if(BUILD_TESTING)
  add_subdirectory(src/test)
endif()

