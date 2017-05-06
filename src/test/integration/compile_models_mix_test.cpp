#include <gtest/gtest.h>

TEST(lang, compile_models_mix) {
  SUCCEED() 
    << "Model compilation done through makefile dependencies." << std::endl
    << "Should have compiled: src/test/test-models/good/*.stan" << std::endl
    << "under mixed mode autodiff" << std::endl;
}
