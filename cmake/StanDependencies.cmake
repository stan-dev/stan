# This file should be included *BEFORE* the stan subdirectory is added.
#
# This requires that STAN_PATH has been set to the path of the stan subdirectory.
#
# In addition, the parent should set(NEED_GTEST ON) and set(NEED_LATEX ON) if
# they are required by the interface for unit tests or building the manual.

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
  include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")
else()
  include_directories(SYSTEM "${STAN_PATH}/lib/eigen_3.2.0")
endif()

# Find Boost
if(USE_SYSTEM_BOOST)
  find_package( Boost 1.54 REQUIRED )
  include_directories(SYSTEM "${Boost_INCLUDE_DIR}")
else()
  include_directories(SYSTEM "${STAN_PATH}/lib/boost_1.54.0")
endif()

# Setup GTest library
if(NEED_GTEST AND NOT HAVE_GTEST)
  if(USE_SYSTEM_GTEST)
    find_package( GTest REQUIRED )
    set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
  else()
    add_subdirectory("${STAN_PATH}/lib/gtest_1.7.0")
  endif()
  set(HAVE_GTEST ON)
endif()

if(NEED_LATEX AND NOT HAVE_LATEX)
  include(UseLATEX)
  set(HAVE_LATEX ON)
endif()



