#include <gtest/gtest.h>

TEST(lang, compile_models_fwd) {
  SUCCEED() 
    << "Model compilation done through makefile dependencies." << std::endl
    << "Should have compiled: src/test/test-models/good/*.stan" << std::endl
    << "under forward mode autodiff" << std::endl;
}
