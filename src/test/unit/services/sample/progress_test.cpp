#include <stan/services/sample/progress.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, progress) {
  std::stringstream out;
  int m;
  int start;
  int finish;
  int refresh;
  bool warmup;
  std::string prefix;
  std::string suffix;

  start = 0;
  finish = 100;
  refresh = 1;
  warmup = true;
  prefix = "";
  suffix = "";

  out.str("");
  m = 0;
  stan::services::sample::progress(m, start, finish, refresh, warmup,
                                   prefix, suffix, out);
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Warmup)", out.str());

  out.str("");
  m = 1;
  prefix = "\r";
  suffix = "\n";
  stan::services::sample::progress(m, start, finish, refresh, warmup,
                                   prefix, suffix, out);
  EXPECT_EQ("\rIteration:  2 / 100 [  2%]  (Warmup)\n", out.str());

  out.str("");
  m = 2;
  warmup = false;
  prefix = "";
  suffix = "";
  stan::services::sample::progress(m, start, finish, refresh, warmup,
                                   prefix, suffix, out);
  EXPECT_EQ("Iteration:  3 / 100 [  3%]  (Sampling)", out.str());
}

