#include <gtest/gtest.h>
#include <stan/prob/distributions/pareto.hpp>

TEST(ProbDistributionsPareto,Pareto) {
  EXPECT_FLOAT_EQ(-1.909543, stan::prob::pareto_log(1.5,0.5,2.0));
  EXPECT_FLOAT_EQ(-25.69865, stan::prob::pareto_log(19.5,0.15,5.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::pareto_log(0.0,0.15,5.0));
}
TEST(ProbDistributionsPareto,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::pareto_log<true>(1.5,0.5,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::pareto_log<true>(19.5,0.15,5.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::pareto_log(0.0,0.15,5.0));
}

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

using stan::prob::pareto_log;

TEST(ProbDistributionsPareto,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double y = 5.0;
  double y_min = 2.0;
  double alpha = 0.5;
  
  EXPECT_NO_THROW(pareto_log(y, y_min, alpha));
  EXPECT_NO_THROW(pareto_log(-inf, y_min, alpha));
  EXPECT_NO_THROW(pareto_log(inf, y_min, alpha));
  EXPECT_NO_THROW(pareto_log(y, y_min, alpha));
  
  EXPECT_THROW(pareto_log(nan, y_min, alpha), std::domain_error);

  EXPECT_THROW(pareto_log(y, nan, alpha), std::domain_error);
  EXPECT_THROW(pareto_log(y, 0.0, alpha), std::domain_error);
  EXPECT_THROW(pareto_log(y, -1.0, alpha), std::domain_error);
  EXPECT_THROW(pareto_log(y, -inf, alpha), std::domain_error);
  EXPECT_THROW(pareto_log(y, inf, alpha), std::domain_error);
  
  EXPECT_THROW(pareto_log(y, y_min, nan), std::domain_error);
  EXPECT_THROW(pareto_log(y, y_min, 0.0), std::domain_error);
  EXPECT_THROW(pareto_log(y, y_min, -1.0), std::domain_error);
  EXPECT_THROW(pareto_log(y, y_min, inf), std::domain_error);
  EXPECT_THROW(pareto_log(y, y_min, -inf), std::domain_error);
}
TEST(ProbDistributionsPareto,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  double y = 5.0;
  double y_min = 2.0;
  double alpha = 0.5;
  
  result = pareto_log(y, y_min, alpha, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = pareto_log(-inf, y_min, alpha, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = pareto_log(inf, y_min, alpha, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = pareto_log(y, y_min, alpha, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = pareto_log(nan, y_min, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  result = pareto_log(y, nan, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, 0.0, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, -1.0, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, -inf, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, inf, alpha, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = pareto_log(y, y_min, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, y_min, 0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, y_min, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, y_min, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = pareto_log(y, y_min, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
