#include <gtest/gtest.h>
#include "stan/prob/distributions/multi_student_t.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

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
  > errno_policy;


TEST(ProbDistributionsMultiStudentT,MultiT) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  double lp = stan::prob::multi_student_t_log(y,nu,mu,Sigma);
  // calc using R's mnormt package's dmt function
  EXPECT_NEAR(-10.1246,lp,0.0001);
}

// FIXME: tests for error conditions
