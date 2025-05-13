#include <stan/run/config_algorithm.hpp>
#include <stan/run/algorithm_type.hpp>
#include <gtest/gtest.h>


TEST(AlgorithmConfigTest, DefaultConstructor) {
  auto config_algorithm = stan::run::config_algorithm::create().build();
  EXPECT_EQ(stan::run::algorithm_t::STAN2_HMC, config_algorithm.algorithm_type());
  EXPECT_EQ(1, config_algorithm.num_chains());
  EXPECT_EQ(1, config_algorithm.random_seed());
}

TEST(AlgorithmConfigTest, Constructor_2) {
  auto config_algorithm = stan::run::config_algorithm::create()
    .algorithm_type(stan::run::algorithm_t::PATHFINDER).build();
  EXPECT_EQ(stan::run::algorithm_t::PATHFINDER, config_algorithm.algorithm_type());
}

TEST(AlgorithmConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto config_algorithm = stan::run::config_algorithm::create()
	       .num_chains(0).build(),
	       std::invalid_argument);
}


TEST(AlgorithmConfigTest, setters_bad) {
  auto config_algorithm = stan::run::config_algorithm::create();
  EXPECT_THROW(config_algorithm.num_chains(0), std::invalid_argument);
}
