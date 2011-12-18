#include "stan/prob/distributions_normal.hpp"
#include "stan/agrad/agrad.hpp"
#include <gtest/gtest.h>

using stan::agrad::var;

class AgradDistributionsPropto : public testing::Test {
  protected:
  virtual void SetUp() {
    y1 = 0.3;
    y2 = 0.6;
    mu = 0.5;
    sigma = 0.5;
    
    expected = stan::prob::normal_log<false>(y1,mu,sigma) 
      - stan::prob::normal_log<false>(y2,mu,sigma);
  }
  var y1;
  var y2;
  var mu;
  var sigma;
  var expected;
  var result;
};

TEST_F(AgradDistributionsPropto,Normal) {
  result = stan::prob::normal_log<true>(y1,mu,sigma)
    - stan::prob::normal_log<true>(y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalY) {
  result = stan::prob::normal_log<true>(y1.val(),mu,sigma)
    - stan::prob::normal_log<true>(y2.val(),mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalMu) {
  result = stan::prob::normal_log<true>(y1,mu.val(),sigma)
    - stan::prob::normal_log<true>(y2,mu.val(),sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalSigma) {
  result = stan::prob::normal_log<true>(y1,mu,sigma.val())
    - stan::prob::normal_log<true>(y2,mu,sigma.val());
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
