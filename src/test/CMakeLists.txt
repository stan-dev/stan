include(StanTesting)

include_directories(SYSTEM "${GTEST_INCLUDE_DIRS}")

# Disable optimization for building tests
foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  string(REGEX REPLACE "-O[^0]" "-O0" ${flag_var} "${${flag_var}}")
  string(REGEX REPLACE "/O[^0]" "/O0" ${flag_var} "${${flag_var}}")
endforeach(flag_var)

add_subdirectory(test-models)

set(TEST_DIRS integration math 
              unit unit-agrad-fwd  unit-agrad-rev)

foreach(curr_test ${TEST_DIRS})
  set(testdir ${CMAKE_CURRENT_SOURCE_DIR}/${curr_test})
  file( GLOB_RECURSE TEST_SRCS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        ${testdir}/*_test.cpp )
  foreach(source ${TEST_SRCS})
    string(REPLACE ".cpp" "${empty_string}" executable ${source})
    string(REPLACE "/"  "_" executable ${executable})
    string(REPLACE "\\"  "_" executable ${executable})
    add_executable(${executable} ${source})
    # A small number of tests rely on generated code, add this dependency to
    # ensure it is generated before trying to build the executable
    add_dependencies(${executable} Test_NoMain_Models)
    target_link_libraries(${executable} stan stanc ${GTEST_LIBRARIES})
    add_test(test_${curr_test}
             ${executable} --gtest_output="xml:${executable}.xml")
  endforeach()
endforeach()

add_subdirectory(unit-distribution)

