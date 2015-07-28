#include <stan/interface_callbacks/var_context_factory/var_context_factory.hpp>
#include <stan/interface_callbacks/var_context_factory/dump_factory.hpp>
#include <gtest/gtest.h>

// FIXME: move to CmdStan
TEST(CmdStan, dump_factory_constructor) {
  stan::interface_callbacks::var_context_factory::dump_factory f;
  
  SUCCEED() 
    << "dump_factory was instantiated properly";
}

TEST(CmdStan, dump_factory_source) {
  stan::interface_callbacks::var_context_factory::dump_factory f;
  
  EXPECT_THROW(f(""),
               std::runtime_error);
  
  stan::io::dump context = f("src/test/unit/interface_callbacks/var_context_factory/var_context_factory.data.R");
  
  EXPECT_TRUE(context.contains_r("a"));
  EXPECT_TRUE(context.contains_r("b"));
  EXPECT_TRUE(context.contains_r("c"));
  EXPECT_FALSE(context.contains_r("d"));
}

TEST(StanInterface, var_context_factory) {
  //stan::interface_callbacks::var_context_factory f;
  SUCCEED() 
    << "Can't instantiate a var_context_factory because it's an abstract class";
}
