#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_student_t.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

using stan::prob::multi_student_t_log;

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
  double lp = multi_student_t_log(y,nu,mu,Sigma);
  // calc using R's mnormt package's dmt function
  EXPECT_NEAR(-10.1246,lp,0.0001);
}

TEST(ProbDistributionsMultiStudentT,DefaultPolicySigma) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  Sigma(0,1) = 10; // non-symmetric
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);
}


TEST(ProbDistributionsMultiStudentT,ErrNoPolicySigma) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  Sigma(0,1) = 10; // non-symmetric
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}


TEST(ProbDistributionsMultiStudentT,DefaultPolicyMu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,ErrNoPolicyMu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));

  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}

TEST(ProbDistributionsMultiStudentT,DefaultPolicyY) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  y(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  y(0) = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));

  y(0) = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
}
TEST(ProbDistributionsMultiStudentT,ErrNoPolicyY) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  y(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));

  y(0) = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));

  y(0) = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}

TEST(ProbDistributionsMultiStudentT,DefaultPolicyNu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  nu = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = 0.0;
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = -1.0;
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  // nu = infinity OK
  nu = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
}


TEST(ProbDistributionsMultiStudentT,PolicySize1) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}
TEST(ProbDistributionsMultiStudentT,ErrnoPolicySize1) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}

TEST(ProbDistributionsMultiStudentT,PolicySize2) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,2);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}
TEST(ProbDistributionsMultiStudentT,ErrnoPolicySize2) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,2);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}

TEST(ProbDistributionsMultiStudentT,PolicySize3) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}
TEST(ProbDistributionsMultiStudentT,ErrnoPolicySize3) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma,errno_policy()));
}
TEST(ProbDistributionsMultiStudentT,ProptoAllDoublesZero) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_FLOAT_EQ(0.0,multi_student_t_log<true>(y,nu,mu,Sigma));
  EXPECT_FLOAT_EQ(0.0,multi_student_t_log<true>(y,nu,mu,Sigma,errno_policy()));

}
