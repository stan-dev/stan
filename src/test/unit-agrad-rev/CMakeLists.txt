file( GLOB_RECURSE TEST_SRCS
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
      *_test.cpp )
foreach(source ${TEST_SRCS})
  add_stan_gtest(UnitAgradRevTests "${source}")
endforeach()

