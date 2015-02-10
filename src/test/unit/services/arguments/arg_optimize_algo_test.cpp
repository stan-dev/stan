#include <gtest/gtest.h>
#include <stan/services/arguments/arg_optimize_algo.hpp>

TEST(StanServicesArguments, arg_optimize_algo) {
  stan::services::arg_optimize_algo arg;

  EXPECT_EQ("algorithm", arg.name());
  EXPECT_EQ("Optimization algorithm", arg.description());

  ASSERT_EQ(3U, arg.values().size());
  EXPECT_EQ("bfgs", arg.values()[0]->name());
  EXPECT_EQ("lbfgs", arg.values()[1]->name());
  EXPECT_EQ("newton", arg.values()[2]->name());
}

