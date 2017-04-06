#include <stan/mcmc/windowed_adaptation.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

TEST(McmcWindowedAdaptation, set_window_params) {
  std::stringstream ss;
  stan::callbacks::stream_writer writer(ss);
  
  stan::mcmc::windowed_adaptation adapter("test");
  
  adapter.set_window_params(10, 1, 1, 1, writer);
  std::string warn_output1 =   std::string("WARNING: No test estimation is\n")
                             + "         performed for num_warmup < 20\n\n";
  EXPECT_EQ(warn_output1, ss.str());
  
  ss.str(std::string());
  
  adapter.set_window_params(100, 75, 50, 25, writer);
  std::string warn_output2 =
    std::string("WARNING: There aren't enough warmup iterations to fit the\n")
    + "         three stages of adaptation as currently configured.\n"
    + "         Reducing each adaptation stage to 15%/75%/10% of\n"
    + "         the given number of warmup iterations:\n"
    + "           init_buffer = 15\n"
    + "           adapt_window = 75\n"
    + "           term_buffer = 10\n\n";
  EXPECT_EQ(warn_output2, ss.str());
 
  ss.str(std::string());
  
  adapter.set_window_params(1000, 75, 50, 25, writer);
  std::string warn_output3;
  EXPECT_EQ(warn_output3, ss.str());
}
