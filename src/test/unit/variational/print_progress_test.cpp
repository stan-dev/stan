#include <stan/variational/print_progress.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, progress) {
  std::stringstream out;
  stan::callbacks::stream_writer writer(out);
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
  m = 1;
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, writer);
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Adaptation)\n", out.str());

  out.str("");
  m = 2;
  prefix = "\r";
  suffix = "\n";
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, writer);
  EXPECT_EQ("\rIteration:  2 / 100 [  2%]  (Adaptation)\n\n", out.str());

  out.str("");
  m = 3;
  warmup = false;
  prefix = "";
  suffix = "";
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, writer);
  EXPECT_EQ("Iteration:  3 / 100 [  3%]  (Variational Inference)\n", out.str());
}

