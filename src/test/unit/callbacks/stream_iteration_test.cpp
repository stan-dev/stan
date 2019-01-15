#include <gtest/gtest.h>
#include <stan/callbacks/stream_iteration.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <iostream>

TEST(StanCallbacks, should_print) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::stream_iteration iteration(logger, 1000, 2000, 1);

  EXPECT_TRUE(iteration.should_print(1));
  EXPECT_TRUE(iteration.should_print(2));
}

TEST(StanCallbacks, stream_iteration) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::stream_iteration iteration(logger, 1000, 2000, 1);

  EXPECT_NO_THROW(iteration(1));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration:    1 / 2000 [  0%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");
  
  EXPECT_NO_THROW(iteration(999));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration:  999 / 2000 [ 49%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(1001));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 1001 / 2000 [ 50%]  (Sampling)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(1999));
  EXPECT_EQ("Iteration: 1999 / 2000 [ 99%]  (Sampling)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(2000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 2000 / 2000 [100%]  (Sampling)\n", info.str());
  other.str("");
  info.str("");

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", info.str());
  other.str("");
  info.str("");
}

TEST(StanCallbacks, stream_iteration_refresh) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::stream_iteration iteration(logger, 1000, 2000, 100);

  for (int n = 1; n <= 2000; ++n) {
    other.str("");
    info.str("");
    EXPECT_NO_THROW(iteration(n));
    if (n % 100 == 0 || n == 1 || n == 2000)
      EXPECT_NE(std::string::npos, info.str().find("Iteration:")) << "iteration " << n;
    else
      EXPECT_EQ("", info.str()) << "iteration " << n;
    EXPECT_EQ("", other.str());
  }
}

TEST(StanCallbacks, stream_iteration_no_print) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::stream_iteration iteration(logger, 1000, 2000, 0);

  for (int n = 1; n <= 2000; ++n) {
    other.str("");
    info.str("");
    EXPECT_NO_THROW(iteration(n));
    EXPECT_EQ("", info.str()) << "iteration " << n;
    EXPECT_EQ("", other.str());
  }
}
