#include <stan/services/sample/progress.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, progress) {
  int m;
  int start;
  int finish;
  int refresh;
  bool warmup;
  std::string out;

  start = 0;
  finish = 100;
  refresh = 1;
  warmup = true;

  m = 0;
  out = stan::services::sample::progress(m, start, finish, refresh, warmup);
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Warmup)", out);

  m = 1;
  out = stan::services::sample::progress(m, start, finish, refresh, warmup);

  EXPECT_EQ("Iteration:  2 / 100 [  2%]  (Warmup)", out);

  m = 2;
  warmup = false;
  out = stan::services::sample::progress(m, start, finish, refresh, warmup);
  EXPECT_EQ("Iteration:  3 / 100 [  3%]  (Sampling)", out);
}

