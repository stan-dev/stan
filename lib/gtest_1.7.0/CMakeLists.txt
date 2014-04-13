# FIXME: we should grab the CMakeLists.txt from the original gtest source
# distribution, but this will suffice for now.

# Find threading library
find_package( Threads )

if(NOT CMAKE_USE_PTHREADS_INIT)
  add_definitions(-DGTEST_HAS_PTHREAD=0)
else()
  add_definitions(-DGTEST_HAS_PTHREAD=1)
endif()
add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

set(GTEST_SRCS 
    src/gtest-all.cc
    src/gtest_main.cc)

include_directories(include .)

add_library(gtest STATIC ${GTEST_SRCS})

set(GTEST_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PARENT_SCOPE)

if(NOT CMAKE_USE_PTHREADS_INIT)
  set(GTEST_LIBRARIES gtest PARENT_SCOPE)
else()
  set(GTEST_LIBRARIES gtest ${CMAKE_THREAD_LIBS_INIT} PARENT_SCOPE)
endif()

