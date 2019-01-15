#include <gtest/gtest.h>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/log_timed_iteration.hpp>
#include <iostream>
#include <chrono>
#include <thread>

TEST(StanCallbacks, log_timed_iteration_with_short_refresh) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::log_timed_iteration iteration(logger, 1000, 2000, 0);

  EXPECT_NO_THROW(iteration(1));
  EXPECT_EQ("Iteration:    1 / 2000 [  0%]  (Warmup)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");

  EXPECT_NO_THROW(iteration(999));
  EXPECT_EQ("Iteration:  999 / 2000 [ 49%]  (Warmup)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");

  EXPECT_NO_THROW(iteration(1001));
  EXPECT_EQ("Iteration: 1001 / 2000 [ 50%]  (Sampling)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");

  EXPECT_NO_THROW(iteration(1999));
  EXPECT_EQ("Iteration: 1999 / 2000 [ 99%]  (Sampling)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");

  EXPECT_NO_THROW(iteration(2000));
  EXPECT_EQ("Iteration: 2000 / 2000 [100%]  (Sampling)\n", info.str());
  EXPECT_EQ("", other.str());
  info.str("");
  other.str("");
}

TEST(StanCallbacks, log_timed_iteration_with_logger_refresh) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  const long milliseconds = 1;
  const double seconds = static_cast<double>(milliseconds) / 1000;
  stan::callbacks::log_timed_iteration iteration(logger, 1000, 2000, seconds);

  EXPECT_NO_THROW(iteration(1));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration:    1 / 2000 [  0%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(999));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("", info.str());
  other.str("");
  info.str("");

  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds + 5));

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(1001));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(2000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("", info.str());
  other.str("");
  info.str("");
}
