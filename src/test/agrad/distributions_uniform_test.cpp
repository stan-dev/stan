#include <gtest/gtest.h>
#include "stan/prob/distributions_uniform.hpp"
#include "stan/agrad/agrad.hpp"

using stan::agrad::var;

template<typename T_y, typename T_lb, typename T_ub, bool propto=true>
var getResult(T_y y1, T_y y2, T_lb lb, T_ub ub) {
  return stan::prob::uniform_log<propto,T_y,T_lb,T_ub>(y1,lb,ub)  
    - stan::prob::uniform_log<propto,T_y,T_lb,T_ub>(y2,lb,ub);
}

class AgradDistributionsPropto : public testing::Test {
  protected:
  virtual void SetUp() {
    y1 = 0.3;
    y2 = 5.0;
    lb = 0.5;
    ub = 11.0;
    
    expected = getResult<var,var,var,false>(y1,y2,lb,ub);
  }
  double y1;
  double y2;
  double lb;
  double ub;
  var expected;
  var result;
};
TEST_F(AgradDistributionsPropto,Uniform) {
  result = getResult<var,var,var>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformY) {
  result = getResult<double,var,var>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformYLb) {
  result = getResult<double,double,var>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformYUb) {
  result = getResult<double,var,double>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformLb) {
  result = getResult<var,double,var>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformLbUb) {
  result = getResult<var,double,double>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
TEST_F(AgradDistributionsPropto,UniformUb) {
  result = getResult<var,var,double>(y1,y2,lb,ub);
  EXPECT_FLOAT_EQ(expected.val(),result.val());
}
