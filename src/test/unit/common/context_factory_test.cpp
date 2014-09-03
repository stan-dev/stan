#include <stan/common/context_factory.hpp>
#include <gtest/gtest.h>

// FIXME: move to CmdStan
TEST(CmdStan, dump_factory_constructor) {
  stan::common::dump_factory f;
  
  SUCCEED() 
    << "dump_factory was instantiated properly";
}

TEST(CmdStan, dump_factory_source) {
  stan::common::dump_factory f;
  
  EXPECT_THROW(f(""),
               std::runtime_error);
  
  stan::io::dump context = f("src/test/unit/common/context_factory.data.R");
  
  EXPECT_TRUE(context.contains_r("a"));
  EXPECT_TRUE(context.contains_r("b"));
  EXPECT_TRUE(context.contains_r("c"));
  EXPECT_FALSE(context.contains_r("d"));
}

TEST(StanCommon, var_context_factory) {
  //stan::common::var_context_factory f;
  SUCCEED() 
    << "Can't instantiate a var_context_factory because it's an abstract class";
}
