#include <gtest/gtest.h>

TEST(lang,compile_standalone_functions) {
  SUCCEED() 
    << "Standalone function compilation done through makefile dependencies." 
    << std::endl
    << "Should have compiled: "
    << "src/test/test-models/good-standalone-functions/*.stanfuncs";
}
