#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/student_t.hpp>

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

using stan::prob::student_t_log;


TEST(ProbDistributionsStudentT,StudentT) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::student_t_log(1.0,1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.596843, stan::prob::student_t_log(-3.0,2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.531024, stan::prob::student_t_log(2.0,1.0,0.0,2.0));
  // need test with scale != 1
}
TEST(ProbDistributionsStudentT,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(1.0,1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(-3.0,2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(2.0,1.0,0.0,2.0));
}
TEST(ProbDistributionsStudentT,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  double y = 1.0;
  double nu = 20.0;
  double mu = 13.0;
  double sigma = 5.0;

  EXPECT_NO_THROW(student_t_log(y, nu, mu, sigma));
  EXPECT_NO_THROW(student_t_log(inf, nu, mu, sigma));
  EXPECT_NO_THROW(student_t_log(-inf, nu, mu, sigma));
  
  EXPECT_THROW(student_t_log(nan, nu, mu, sigma), std::domain_error);
  
  EXPECT_THROW(student_t_log(y, 0.0, mu, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, -1.0, mu, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, inf, mu, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, -inf, mu, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, nan, mu, sigma), std::domain_error);
  
  EXPECT_THROW(student_t_log(y, nu, inf, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, -inf, sigma), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, nan, sigma), std::domain_error);

  EXPECT_THROW(student_t_log(y, nu, mu, 0.0), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, mu, -1.0), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, mu, inf), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, mu, -inf), std::domain_error);
  EXPECT_THROW(student_t_log(y, nu, inf, nan), std::domain_error);
}
TEST(ProbDistributionsStudentT,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  double y = 1.0;
  double nu = 20.0;
  double mu = 13.0;
  double sigma = 5.0;
  double result;
  
  result = student_t_log(y, nu, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = student_t_log(inf, nu, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = student_t_log(-inf, nu, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = student_t_log(nan, nu, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = student_t_log(y, 0.0, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, -1.0, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, -inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nan, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = student_t_log(y, nu, inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, -inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, nan, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  result = student_t_log(y, nu, mu, 0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, mu, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, mu, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, mu, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = student_t_log(y, nu, inf, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
