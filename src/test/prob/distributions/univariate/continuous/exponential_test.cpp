#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>


TEST(ProbDistributionsExponential,Cumulative) {
  using std::numeric_limits;
  using stan::prob::exponential_cdf;
  EXPECT_FLOAT_EQ(0.95021293, exponential_cdf(2.0,1.5));
  EXPECT_FLOAT_EQ(1.0, exponential_cdf(15.0,3.9));
  EXPECT_FLOAT_EQ(0.62280765, exponential_cdf(0.25,3.9));

  // ??
  // EXPECT_FLOAT_EQ(0.0, 
  //                 exponential_cdf(-numeric_limits<double>::infinity(),
  //                                 1.5));
  EXPECT_FLOAT_EQ(0.0, exponential_cdf(0.0,1.5));
  EXPECT_FLOAT_EQ(1.0, 
                  exponential_cdf(numeric_limits<double>::infinity(),
                                  1.5));

}
