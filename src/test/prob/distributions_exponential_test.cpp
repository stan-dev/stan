#include <gtest/gtest.h>
#include "stan/prob/distributions_exponential.hpp"

TEST(ProbDistributions,Exponential) {
  EXPECT_FLOAT_EQ(-2.594535, stan::prob::exponential_log(2.0,1.5));
  EXPECT_FLOAT_EQ(-57.13902, stan::prob::exponential_log(15.0,3.9));
}

TEST(ProbDistributionsCumulative,Exponential) {
  // values from R
  EXPECT_FLOAT_EQ(0.95021293, stan::prob::exponential_p(2.0,1.5));
  EXPECT_FLOAT_EQ(0.0, stan::prob::exponential_p(0,1.5));
  EXPECT_FLOAT_EQ(1.0, stan::prob::exponential_p(15.0,3.9));
  EXPECT_FLOAT_EQ(0.62280765, stan::prob::exponential_p(0.25,3.9));
}

TEST(ProbDistributionsTruncated,ExponentialLowHigh) {
  double x, beta, low, high;
  
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 4.0;
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  
  x = 1.5;
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::exponential_trunc_lh_log(x, beta, low, high));

  x = 2.0;
  EXPECT_FLOAT_EQ(0.4565343, stan::prob::exponential_trunc_lh_log(x, beta, low, high));

  x = 3.0;
  EXPECT_FLOAT_EQ(-1.043466, stan::prob::exponential_trunc_lh_log(x, beta, low, high));

  x = 4.0;
  EXPECT_FLOAT_EQ(-2.543466, stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  
  x = 5.0;
  EXPECT_FLOAT_EQ(log (0.0), stan::prob::exponential_trunc_lh_log(x, beta, low, high));
}
TEST(ProbDistributionsTruncated,ExponentialLowHighDefaultPolicyX) {
  double x, beta, low, high;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 5.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  x = 0.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  x = -10.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialLowHighDefaultPolicyBeta) {
  double x, beta, low, high;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 5.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  beta = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  
  beta = 0.0;
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error) <<
    "beta = 0 should throw error";
  
  beta = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error) <<
    "beta < 0 should throw error";

  beta = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  
  beta = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  
  beta = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialLowHighDefaultPolicyLow) {
  double x, beta, low, high;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 5.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  low = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));
  low = 0.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high)) <<
    "low = 0 should not throw an error";

  low = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error) <<
    "low < 0 should throw error";
  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  low = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
}  
TEST(ProbDistributionsTruncated,ExponentialLowHighDefaultPolicyHigh) {
  double x, beta, low, high;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 4.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));

  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  high = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
  high = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error);
}  
TEST(ProbDistributionsTruncated,ExponentialLowHighDefaultPolicyLowHigh) {
  double x, beta, low, high;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  high = 4.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high));

  low = 2.0;
  high = 2.0;
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error)
    << "low == high should throw an exception";
  
  low = 3.0;
  high = 2.0;
  EXPECT_THROW(stan::prob::exponential_trunc_lh_log(x, beta, low, high), std::domain_error)
    << "low > high should throw an exception";
  
}  

TEST(ProbDistributionsTruncated,ExponentialLow) {
  double x, beta, low;
  
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::exponential_trunc_l_log(x, beta, low));
  
  x = 1.5;
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::exponential_trunc_l_log(x, beta, low));

  x = 2.0;
  EXPECT_FLOAT_EQ(0.4054651, stan::prob::exponential_trunc_l_log(x, beta, low));

  x = 3.0;
  EXPECT_FLOAT_EQ(-1.094535, stan::prob::exponential_trunc_l_log(x, beta, low));
  
  x = 5.0;
  EXPECT_FLOAT_EQ(-4.094535, stan::prob::exponential_trunc_l_log(x, beta, low));
}
TEST(ProbDistributionsTruncated,ExponentialLowDefaultPolicyX) {
  double x, beta, low;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  x = 0.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  x = -10.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialLowDefaultPolicyBeta) {
  double x, beta, low;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  beta = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  
  beta = 0.0;
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error) <<
    "beta = 0 should throw error";
  
  beta = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error) <<
    "beta < 0 should throw error";

  beta = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  
  beta = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  
  beta = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialLowDefaultPolicyLow) {
  double x, beta, low;
  x = 1.0;
  beta = 1.5;
  low = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  low = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low));
  low = 0.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_l_log(x, beta, low)) <<
    "low = 0 should not throw an error";

  low = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error) <<
    "low < 0 should throw error";
  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
  low = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_l_log(x, beta, low), std::domain_error);
}  

TEST(ProbDistributionsTruncated,ExponentialHigh) {
  double x, beta, high;
  
  x = 1.0;
  beta = 1.5;
  high = 2.0;
  EXPECT_FLOAT_EQ(-1.043466, stan::prob::exponential_trunc_h_log(x, beta, high));
  
  x = 1.5;
  EXPECT_FLOAT_EQ(-1.793466, stan::prob::exponential_trunc_h_log(x, beta, high));

  x = 2.0;
  EXPECT_FLOAT_EQ(-2.543466, stan::prob::exponential_trunc_h_log(x, beta, high));

  x = 3.0;
  EXPECT_FLOAT_EQ(log (0.0), stan::prob::exponential_trunc_h_log(x, beta, high));
  
  x = 5.0;
  EXPECT_FLOAT_EQ(log (0.0), stan::prob::exponential_trunc_h_log(x, beta, high));
}
TEST(ProbDistributionsTruncated,ExponentialHighDefaultPolicyX) {
  double x, beta, high;
  x = 1.0;
  beta = 1.5;
  high = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  x = 0.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  x = 10.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialHighDefaultPolicyBeta) {
  double x, beta, high;
  x = 1.0;
  beta = 1.5;
  high = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  beta = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  
  beta = 0.0;
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error) <<
    "beta = 0 should throw error";
  
  beta = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error) <<
    "beta < 0 should throw error";

  beta = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  
  beta = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  
  beta = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
}
TEST(ProbDistributionsTruncated,ExponentialHighDefaultPolicyHigh) {
  double x, beta, high;
  x = 1.0;
  beta = 1.5;
  high = 2.0;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));
  high = 0.1;
  EXPECT_NO_THROW(stan::prob::exponential_trunc_h_log(x, beta, high));

  high = 0.0;
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error) <<
    "high = 0 should throw an exception";

  high = -1.0;
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error) <<
    "high < 0 should throw error";
  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  high = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
  high = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::exponential_trunc_h_log(x, beta, high), std::domain_error);
}  


