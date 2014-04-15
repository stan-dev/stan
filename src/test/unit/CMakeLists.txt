
file( GLOB_RECURSE TEST_SRCS
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
      *_test.cpp )
foreach(source ${TEST_SRCS})
  string(REPLACE ".cpp" "${empty_string}" executable ${source})
  string(REPLACE "/"  "_" executable ${executable})
  string(REPLACE "\\"  "_" executable ${executable})
  set(executable unit_${executable})

  add_executable(${executable} ${source})
  # A small number of tests rely on generated code, add this dependency to
  # ensure it is generated before trying to build the executable
#  add_dependencies(${executable} ${NOMAIN_SRCS})
  add_dependencies(${executable} Test_NoMain_Models)
  target_link_libraries(${executable} stan stanc ${GTEST_LIBRARIES})
  add_test(UnitTests_${curr_test}
           ${executable} --gtest_output="xml:${executable}.xml")
endforeach()

