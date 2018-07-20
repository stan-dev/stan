#include <gtest/gtest.h>
#include <stan/callbacks/stream_iteration.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <iostream>

TEST(StanCallbacks, stream_iteration_with_ostream) {
  std::stringstream ostream;
  stan::callbacks::stream_iteration iteration(ostream, 1000, 2000, 1);

  EXPECT_NO_THROW(iteration(1));
  EXPECT_EQ("Iteration:    1 / 2000 [  0%]  (Warmup)\n", ostream.str());
  ostream.str("");


  EXPECT_NO_THROW(iteration(999));
  EXPECT_EQ("Iteration:  999 / 2000 [ 49%]  (Warmup)\n", ostream.str());
  ostream.str("");

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", ostream.str());
  ostream.str("");

  EXPECT_NO_THROW(iteration(1001));
  EXPECT_EQ("Iteration: 1001 / 2000 [ 50%]  (Sampling)\n", ostream.str());
  ostream.str("");

  EXPECT_NO_THROW(iteration(1999));
  EXPECT_EQ("Iteration: 1999 / 2000 [ 99%]  (Sampling)\n", ostream.str());
  ostream.str("");

  EXPECT_NO_THROW(iteration(2000));
  EXPECT_EQ("Iteration: 2000 / 2000 [100%]  (Sampling)\n", ostream.str());
  ostream.str("");
}

TEST(StanCallbacks, stream_iteration_with_ostream_refresh) {
  std::stringstream ostream;
  stan::callbacks::stream_iteration iteration(ostream, 1000, 2000, 100);

  for (int n = 1; n <= 2000; ++n) {
    ostream.str("");
    EXPECT_NO_THROW(iteration(n));
    if (n % 100 == 0 || n == 1 || n == 2000)
      EXPECT_NE(std::string::npos, ostream.str().find("Iteration:")) << "iteration " << n;
    else
      EXPECT_EQ("", ostream.str()) << "iteration " << n;
  }
}


TEST(StanCallbacks, stream_iteration_with_logger) {
  std::stringstream info;
  std::stringstream other;
  stan::callbacks::stream_logger logger(other, info, other, other, other);
  stan::callbacks::stream_iteration iteration(logger, 1000, 2000, 1);

  EXPECT_NO_THROW(iteration(1000));
  EXPECT_EQ("", other.str());
  EXPECT_EQ("Iteration: 1000 / 2000 [ 50%]  (Warmup)\n", info.str());
}
