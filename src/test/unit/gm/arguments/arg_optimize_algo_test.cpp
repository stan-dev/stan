#include <gtest/gtest.h>
#include <stan/gm/arguments/arg_optimize_algo.hpp>

TEST(StanGmArguments, arg_optimize_algo) {
  stan::gm::arg_optimize_algo arg;

  EXPECT_EQ("algorithm", arg.name());
  EXPECT_EQ("Optimization algorithm", arg.description());

  ASSERT_EQ(3, arg.values().size());
  EXPECT_EQ("bfgs", arg.values()[0]->name());
  EXPECT_EQ("lbfgs", arg.values()[1]->name());
  EXPECT_EQ("newton", arg.values()[2]->name());
}

