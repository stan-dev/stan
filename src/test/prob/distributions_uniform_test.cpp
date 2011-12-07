// These tests should not have reference to stan::agrad::var. Distribution tests 
// with stan::agrad::var should be placed in src/test/agrad/distributions_test.cpp

#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "stan/prob/distributions_uniform.hpp"

TEST(ProbDistributions,Uniform) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.2,0.0,1.0)));
  EXPECT_FLOAT_EQ(2.0, exp(stan::prob::uniform_log(0.2,-0.25,0.25)));
  EXPECT_FLOAT_EQ(0.1, exp(stan::prob::uniform_log(101.0,100.0,110.0)));
  // lower boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log(-1.0,0.0,1.0)));
  // upper boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(1.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log(2.0,0.0,1.0)));  
}
TEST(ProbDistributions,UniformDefaultPolicy) {
  // lower bound higher than the upper bound
  EXPECT_THROW (stan::prob::uniform_log(0.0,1.0,0.0), std::domain_error);
  // lower and upper boundary the same 
  EXPECT_THROW (stan::prob::uniform_log(0.0, 0.0, 0.0), std::domain_error);
  EXPECT_THROW (stan::prob::uniform_log(1.0, 0.0, 0.0), std::domain_error);
  EXPECT_THROW (stan::prob::uniform_log(-1.0, 0.0, 0.0), std::domain_error);
}
TEST(ProbDistributions,UniformErrorNoPolicy) {
  using boost::math::policies::policy;
  using boost::math::policies::evaluation_error;
  using boost::math::policies::domain_error;
  using boost::math::policies::overflow_error;
  using boost::math::policies::domain_error;
  using boost::math::policies::pole_error;
  using boost::math::policies::errno_on_error;

  typedef policy<
    domain_error<errno_on_error>, 
    pole_error<errno_on_error>,
    overflow_error<errno_on_error>,
    evaluation_error<errno_on_error> 
    > pol;
  
  double y = 0;
  double result;
  // lower and uppper bounds same
  EXPECT_NO_THROW (result = stan::prob::uniform_log (y, 0.0, 0.0, pol()));
  EXPECT_EQ (33, errno);
  

}
