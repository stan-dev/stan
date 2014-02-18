#include <stan/common/print_progress.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, print_progress_FIXME_redirecting) {
  // The redirect_cout stringstream traps std::cout
  std::stringstream redirect_cout;
  std::streambuf* old_cout_rdbuf = std::cout.rdbuf(redirect_cout.rdbuf());
  
  for (int m = -10; m < 10; m++)
    for (int start = -10; start < 10; start++)
      for (int finish = -10; finish < 10; finish++)
        for (int refresh = -10; refresh < 10; refresh++) {
            bool warmup;
            warmup = true;
            EXPECT_NO_THROW(stan::common::print_progress(m, start, finish, refresh, warmup));
            warmup = false;
            EXPECT_NO_THROW(stan::common::print_progress(m, start, finish, refresh, warmup));
        }
  std::cout.rdbuf(old_cout_rdbuf);
  SUCCEED() 
    << "FIXME: This function needs to take in an output stream";
}
