#include <gtest/gtest.h>

TEST(gm,compile_models) {
  SUCCEED() 
    << "Model compilation done through makefile dependencies." << std::endl
    << "Should have compiled: src/test/test-models/good/*.stan";
}
