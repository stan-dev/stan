include(StanTesting)

include_directories(SYSTEM "${GTEST_INCLUDE_DIRS}")

# Disable optimization for building tests
foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  string(REGEX REPLACE "-O[^0]" "-O0" ${flag_var} "${${flag_var}}")
  string(REGEX REPLACE "/O[^0]" "/O0" ${flag_var} "${${flag_var}}")
endforeach(flag_var)

if (BUILD_STAN_UNIT_TESTS)
  add_subdirectory(test-models)
  add_subdirectory(integration)
  add_subdirectory(unit)
  add_subdirectory(unit-agrad-fwd)
  add_subdirectory(unit-agrad-rev)
endif()

if (BUILD_STAN_DIST_TESTS)
  add_subdirectory(unit-distribution)
endif()


