#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <boost/math/policies/policy.hpp>
#include "stan/prob/distributions.hpp"

TEST(distributions,Normal) {
  // values from R dnorm()
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::normal_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::normal_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::normal_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::normal_log(-3.5,1.9,7.2));
}
TEST(distributions,NormalDefaultPolicy) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
}
TEST(distributions,NormalErrnoPolicy) {
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
    > my_policy;

  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
}

TEST(distributions,NormalPropTo) {
  double diff = stan::prob::normal_propto_log (0.0, 0.0, 1.0) - (-0.9189385);
  
  EXPECT_FLOAT_EQ(-0.9189385 + diff, stan::prob::normal_propto_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939 + diff,  stan::prob::normal_propto_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939 + diff,  stan::prob::normal_propto_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270 + diff,  stan::prob::normal_propto_log(-3.5,1.9,7.2));
}
TEST(distributions,NormalPropToDefaultPolicy) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_propto_log(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_propto_log(0.0,0.0,sigma_d), std::domain_error);
}
TEST(distributions,NormalPropToErronoPolicy) {
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
    > my_policy;

  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_propto_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_propto_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
}
