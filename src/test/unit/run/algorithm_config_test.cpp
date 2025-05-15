#include <stan/run/config_algorithm.hpp>
#include <stan/run/algorithm_type.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>

TEST(AlgorithmConfigTest, DefaultConstructor) {
  auto config_algorithm = stan::run::config_algorithm::create().build();
  EXPECT_EQ(stan::run::algorithm_t::STAN2_HMC, config_algorithm.algorithm_type());
  EXPECT_EQ(1, config_algorithm.num_chains());
  EXPECT_EQ(1, config_algorithm.random_seed());
}

TEST(AlgorithmConfigTest, Constructor_2) {
  std::stringstream logger_stream;
  stan::callbacks::stream_logger
    logger(logger_stream, logger_stream, logger_stream, logger_stream, logger_stream);

  auto config_algorithm = stan::run::config_algorithm::create()
    .algorithm_type(stan::run::algorithm_t::PATHFINDER).logger(&logger).build();
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



TEST(AlgorithmConfigTest, LoggerGetter) {
  // Create a logger
  std::stringstream logger_stream;
  stan::callbacks::stream_logger logger(logger_stream, logger_stream, 
                                      logger_stream, logger_stream, logger_stream);

  // Configure with logger
  auto config = stan::run::config_algorithm::create()
    .logger(&logger)
    .build();
    
  // Test the getter returns the same logger pointer
  EXPECT_EQ(&logger, config.logger());
}

TEST(AlgorithmConfigTest, NullLoggerValid) {
  // Configure with null logger
  auto config = stan::run::config_algorithm::create()
    .logger(nullptr)
    .build();
    
  // Verify null logger is accepted
  EXPECT_EQ(nullptr, config.logger());
  
  // Verify default constructor also sets null logger
  auto default_config = stan::run::config_algorithm::create().build();
  EXPECT_EQ(nullptr, default_config.logger());
}


TEST(AlgorithmConfigTest, LoggerFunctionality) {
  // Create a single stream for simplified testing
  std::stringstream log_stream;
  
  // Create logger with the same stream for all levels
  stan::callbacks::stream_logger logger(log_stream, log_stream, 
                                      log_stream, log_stream, log_stream);

  // Configure with logger
  auto config = stan::run::config_algorithm::create()
    .logger(&logger)
    .build();
    
  // FIRST TEST: Verify the logger pointer is correctly stored
  EXPECT_EQ(&logger, config.logger());
  
  // SECOND TEST: Verify direct logger usage works
  logger.info("Direct info message");
  std::string stream_content = log_stream.str();
  std::cout << "Direct logger output: '" << stream_content << "'" << std::endl;
  EXPECT_TRUE(stream_content.find("Direct info message") != std::string::npos);
  
  // Clear the stream for the next test
  log_stream.str("");
  
  // THIRD TEST: Now test through the config object
  config.logger()->info("Config info message");
  stream_content = log_stream.str();
  std::cout << "Config logger output: '" << stream_content << "'" << std::endl;
  EXPECT_TRUE(stream_content.find("Config info message") != std::string::npos);
}
