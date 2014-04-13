
# Build utility which generates the tests
add_executable( generate_tests generate_tests.cpp )

# Find the headers for the generated unit tests
file(GLOB_RECURSE DIST_TEST_HDRS 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *_test.hpp)

function(count_tests filename countvar)
  file(STRINGS ${filename} contents LIMIT_COUNT 1)
  string(REPLACE "// Arguments: " "${empty_string}" contents ${contents})
  string(REGEX REPLACE "[ ,]+" ";" contents "${contents}")

  set(count 1)
  foreach(type ${contents})
    string(TOLOWER ${type} type)
    if(${type} STREQUAL "int")
      math(EXPR count "${count}*1")
    elseif(${type} STREQUAL "double")
      math(EXPR count "${count}*2")
    elseif(${type} STREQUAL "ints")
      math(EXPR count "${count}*4")
    elseif(${type} STREQUAL "doubles")
      math(EXPR count "${count}*8")
    else()
      message(ERROR "Unrecognized type: ${type} in test header ${filename}")
    endif()
  endforeach()
  math(EXPR count "(${count}/100) - ((${count}/128)%1) + 1")
  set(${countvar} ${count} PARENT_SCOPE)
endfunction(count_tests)

function(gen_test_srcs hdrname numtests srcvar)
  math(EXPR test_range "${numtests} - 1")
  foreach(testnum RANGE 0 ${numtests})
    if(testnum LESS 10)
      string(REPLACE "_test.hpp" "_0000${testnum}_generated_test.cpp" curr_src ${hdrname})
    elseif(testnum LESS 100)
      string(REPLACE "_test.hpp" "_000${testnum}_generated_test.cpp" curr_src ${hdrname})
    elseif(testnum LESS 1000)
      string(REPLACE "_test.hpp" "_00${testnum}_generated_test.cpp" curr_src ${hdrname})
    elseif(testnum LESS 10000)
      string(REPLACE "_test.hpp" "_0${testnum}_generated_test.cpp" curr_src ${hdrname})
    else()
      string(REPLACE "_test.hpp" "_${testnum}_generated_test.cpp" curr_src ${hdrname})
    endif()
    list(APPEND SRCS "${curr_src}")
  endforeach()
  set(${srcvar} ${SRCS} PARENT_SCOPE)
endfunction(gen_test_srcs)

include_directories(${CURRENT_SOURCE_DIR})

# Generate distribution unit tests
foreach(curr_hdr ${DIST_TEST_HDRS})
  string(REPLACE "/"  "_" dist_name ${curr_hdr})
  string(REPLACE "_test.hpp"  "${empty_string}" dist_name ${dist_name})

  # Count how many tests there are
  count_tests(${CMAKE_CURRENT_SOURCE_DIR}/${curr_hdr} numtests)

  # For each test, generate the source file name
  gen_test_srcs(${curr_hdr} ${numtests} curr_srcs)

  # Copy the header file to the build directory if it's different
  if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR}))
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${curr_hdr}
                   ${CMAKE_CURRENT_BINARY_DIR}/${curr_hdr} COPYONLY)
  endif()

  # Add a custom command to build the test generation program
  add_custom_command(OUTPUT ${curr_srcs}
                     COMMAND generate_tests ${curr_hdr}
                     DEPENDS generate_tests)

  foreach(curr_src ${curr_srcs})
    string(REPLACE ".cpp" "${empty_string}" executable ${curr_src})

    add_executable(${executable} ${curr_src})
    target_link_libraries(${executable} stan stanc ${GTEST_LIBRARIES})
    add_test(test_unit_dist_gen_${dist_name}
             ${executable} --gtest_output="xml:${executable}.xml")
  endforeach()
endforeach()



