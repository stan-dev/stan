#include <stan/gm/command.hpp>
#include <gtest/gtest.h>
#include <sstream> 

TEST(GmCommand, do_print_2_args) {
  int refresh;
  
  refresh = 0;
  for (int n = -10; n <= 10; n++) {
    EXPECT_FALSE(stan::gm::do_print(n, refresh))
      << "should return false for refresh == 0.  "
      << "called with n=" << n 
      << ", refresh=" << refresh;
  }

  refresh = 10;
  for (int n = 0; n <= 30; n++) {
    if (n == 0 || n == 9 || n == 19 || n == 29) {
      EXPECT_TRUE(stan::gm::do_print(n, refresh))
        << "should return true.  "
        << "called with n=" << n 
        << ", refresh=" << refresh;
    } else {
      EXPECT_FALSE(stan::gm::do_print(n, refresh))
        << "should return false.  "
        << "called with n=" << n 
        << ", refresh=" << refresh;
    }
  }
}

TEST(GmCommand, do_print_3_args) {
  int refresh;
  bool special;

  refresh = 0;
  special = false;
  for (int n = -10; n <= 10; n++) {
    EXPECT_FALSE(stan::gm::do_print(n, special, refresh))
      << "should return false for refresh == 0.  "
      << "called with n=" << n 
      << ", special=" << special
      << ", refresh=" << refresh;
  }

  refresh = 0;
  special = true;
  for (int n = -10; n <= 10; n++) {
    EXPECT_FALSE(stan::gm::do_print(n, special, refresh))
      << "should return false for refresh == 0.  "
      << "called with n=" << n 
      << ", special=" << special
      << ", refresh=" << refresh;
  }


  refresh = 10;
  special = false;
  for (int n = 0; n <= 30; n++) {
    EXPECT_EQ(stan::gm::do_print(n, refresh),
              stan::gm::do_print(n, special, refresh))
      << "should return the same result as the 2 arg version when special==false.  " 
      << "called with n=" << n 
      << ", special=" << special
      << ", refresh=" << refresh;
  }

  refresh = 10;
  special = true;
  for (int n = 0; n <= 30; n++) {
    EXPECT_TRUE(stan::gm::do_print(n, special, refresh))
      << "should return true when special == true.  "
      << "called with n=" << n 
      << ", special=" << special
      << ", refresh=" << refresh;
  }

}

TEST(GmCommand, write_iteration_csv) {
  std::stringstream stream;
  double lp;
  std::vector<double> model_values;

  stream.str("");
  lp = 0.0;
  model_values.clear();
  stan::gm::write_iteration_csv(stream, lp, model_values);
  EXPECT_EQ("0\n", stream.str());

  stream.str("");
  lp = 1.0;
  model_values.clear();
  model_values.push_back(2.0);
  model_values.push_back(3.0);
  stan::gm::write_iteration_csv(stream, lp, model_values);
  EXPECT_EQ("1,2,3\n", stream.str());
}
