#include <gtest/gtest.h>
#include "stan/prob/distributions_normal.hpp"
#include "stan/agrad/agrad.hpp"


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
  double y1;
  double y2;
  double mu;
  double sigma;
  var expected;
  var result;
};

template<typename T_y, typename T_loc, typename T_scale>
var getResult(T_y y1, T_y y2, T_loc mu, T_scale sigma) {
  return stan::prob::normal_log<true,T_y,T_loc,T_scale>(y1,mu,sigma)  
    - stan::prob::normal_log<true,T_y,T_loc,T_scale>(y2,mu,sigma);
}


TEST_F(AgradDistributionsPropto,Normal) {
  result = getResult<var,var,var>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalY) {
  result = getResult<double,var,var>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalYMu) {
  result = getResult<double,double,var>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalYSigma) {
  result = getResult<double,var,double>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalMu) {
  result = getResult<var,double,var>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalMuSigma) {
  result = getResult<var,double,double>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,NormalSigma) {
  result = getResult<var,var,double>(y1,y2,mu,sigma);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
