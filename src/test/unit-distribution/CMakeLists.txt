# Build utility which generates the tests
add_executable( generate_tests generate_tests.cpp )

# Find the headers for the generated unit tests
file(GLOB_RECURSE DIST_TEST_HDRS 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *_test.hpp)

# Generated unit tests include their header files relative to this directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Generate distribution unit tests
foreach(curr_hdr ${DIST_TEST_HDRS})
  string(REPLACE "/"  "_" dist_name ${curr_hdr})
  string(REPLACE "\\"  "_" dist_name ${dist_name})
  string(REPLACE "_test.hpp"  "${empty_string}" dist_name ${dist_name})

  # Count how many tests there are
  count_tests(${CMAKE_CURRENT_SOURCE_DIR}/${curr_hdr} numtests)

  # For each test, generate the source file name
  gen_test_srcs(${curr_hdr} ${numtests} curr_srcs)

  # Copy the header file to the build directory if it's different
#  if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR}))
#    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${curr_hdr}
#                   ${CMAKE_CURRENT_BINARY_DIR}/${curr_hdr} COPYONLY)
#  endif()

  # Add a custom command to build the test generation program
  add_custom_command(OUTPUT ${curr_srcs}
                     COMMAND ${RUNCMD} generate_tests ${curr_hdr}
                     DEPENDS generate_tests ${curr_hdr})

  foreach(curr_src ${curr_srcs})
    add_stan_gtest(UnitDistributionTests "${curr_src}")
  endforeach()
endforeach()



