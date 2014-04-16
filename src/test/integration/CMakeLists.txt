# FIXME: The compiled_models integration test isn't working right yet as 
# compiled_models_test.cpp is basically a no-op.  Need to do something
# different here.

file( GLOB_RECURSE TEST_SRCS
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
      *_test.cpp )
foreach(source ${TEST_SRCS})
  string(REPLACE ".cpp" "${empty_string}" executable ${source})
  string(REPLACE "/"  "_" executable ${executable})
  string(REPLACE "\\"  "_" executable ${executable})
  set(executable integration_${executable})
  add_executable(${executable} ${source})
  add_dependencies(${executable} Test_SyntaxOnly_Models)
  target_link_libraries(${executable} stan stanc ${GTEST_LIBRARIES})
  add_test(IntegrationTests
           ${RUNCMD} ${CMAKE_CURRENT_BINARY_DIR}/${executable} --gtest_output="xml:${executable}.xml")
endforeach()

