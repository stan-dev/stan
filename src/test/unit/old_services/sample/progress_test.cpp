#include <stan/old_services/sample/progress.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, progress) {
  std::string out;
  int m;
  int start;
  int finish;
  int refresh;
  bool warmup;

  start = 0;
  finish = 100;
  refresh = 1;
  warmup = true;

  m = 0;
  out = stan::services::sample::progress(m, start, finish, refresh, warmup);
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Warmup)", out);
}

