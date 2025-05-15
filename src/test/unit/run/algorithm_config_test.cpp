#include <stan/run/config_algorithm.hpp>
#include <stan/run/algorithm_type.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>


class AlgorithmConfigTest : public testing::Test {
protected:
  void SetUp() override {
    // Create the logger that writes to all streams
    logger = std::make_unique<stan::callbacks::stream_logger>(
        logger_stream, logger_stream, logger_stream, logger_stream, logger_stream);
  }

  std::stringstream logger_stream;
  std::unique_ptr<stan::callbacks::stream_logger> logger;
};


TEST_F(AlgorithmConfigTest, DefaultConstructor) {
  auto config_algorithm = stan::run::config_algorithm::create().build();
  EXPECT_EQ(stan::run::algorithm_t::STAN2_HMC, config_algorithm.algorithm_type());
  EXPECT_EQ(1, config_algorithm.num_chains());
  EXPECT_EQ(1, config_algorithm.random_seed());
}

TEST_F(AlgorithmConfigTest, Constructor_2) {
  auto config_algorithm = stan::run::config_algorithm::create()
    .algorithm_type(stan::run::algorithm_t::PATHFINDER).logger(logger).build();
  EXPECT_EQ(stan::run::algorithm_t::PATHFINDER, config_algorithm.algorithm_type());
  EXPECT_EQ(logger.get(), config_algorithm.logger());
}

TEST_F(AlgorithmConfigTest, Constructor_bad) {
  EXPECT_THROW(
	       auto config_algorithm = stan::run::config_algorithm::create()
	       .num_chains(0).build(),
	       std::invalid_argument);
}


TEST_F(AlgorithmConfigTest, setters_bad) {
  auto config_algorithm = stan::run::config_algorithm::create();
  EXPECT_THROW(config_algorithm.num_chains(0), std::invalid_argument);
}
