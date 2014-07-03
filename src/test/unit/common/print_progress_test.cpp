#include <stan/common/print_progress.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, print_progress) {
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
  stan::common::print_progress(m, start, finish, refresh, warmup,
                               prefix, suffix, out);
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Warmup)", out.str());

  out.str("");
  m = 1;
  prefix = "\r";
  suffix = "\n";
  stan::common::print_progress(m, start, finish, refresh, warmup,
                               prefix, suffix, out);
  EXPECT_EQ("\rIteration:  2 / 100 [  2%]  (Warmup)\n", out.str());

  out.str("");
  m = 2;
  warmup = false;
  prefix = "";
  suffix = "";
  stan::common::print_progress(m, start, finish, refresh, warmup,
                               prefix, suffix, out);
  EXPECT_EQ("Iteration:  3 / 100 [  3%]  (Sampling)", out.str());
}

