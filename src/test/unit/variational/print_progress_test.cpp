#include <stan/variational/print_progress.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, progress) {
  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);
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

  debug.str("");
  info.str("");
  warn.str("");
  error.str("");
  fatal.str("");
  m = 1;
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, logger);
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("Iteration:  1 / 100 [  1%]  (Adaptation)\n", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());

  debug.str("");
  info.str("");
  warn.str("");
  error.str("");
  fatal.str("");
  m = 2;
  prefix = "\r";
  suffix = "\n";
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, logger);
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("\rIteration:  2 / 100 [  2%]  (Adaptation)\n\n", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());

  debug.str("");
  info.str("");
  warn.str("");
  error.str("");
  fatal.str("");
  m = 3;
  warmup = false;
  prefix = "";
  suffix = "";
  stan::variational::print_progress(m, start, finish, refresh, warmup,
                                    prefix, suffix, logger);
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("Iteration:  3 / 100 [  3%]  (Variational Inference)\n", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}
