#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/discrete/hypergeometric.hpp>

using std::vector;
using stan::prob::hypergeometric_log;

TEST(ProbDistributionsHypergeometric,Hypergeometric) {
  EXPECT_FLOAT_EQ(-4.119424, stan::prob::hypergeometric_log(5,15,10,10));
  EXPECT_FLOAT_EQ(-2.302585, stan::prob::hypergeometric_log(0,2,3,2));
}
TEST(ProbDistributionsHypergeometric,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::hypergeometric_log<true>(5,15,10,10));
  EXPECT_FLOAT_EQ(0.0, stan::prob::hypergeometric_log<true>(0,2,3,2));
}
TEST(ProbDistributionsHypergeometric,iiiv) {
  int n = 5;
  int N = 15;
  int a = 10;
  vector<int> b(2);
  b[0] = 10;
  b[1] = 15;
  
  double expected_logp = hypergeometric_log(n, N, a, b[0]) + hypergeometric_log(n, N, a, b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,iivi) {
  int n = 5;
  int N = 15;
  vector<int> a(2);
  a[0] = 10;
  a[1] = 8;
  int b = 15;
  
  double expected_logp = hypergeometric_log(n, N, a[0], b) + hypergeometric_log(n, N, a[1], b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,iivv) {
  int n = 5;
  int N = 15;
  vector<int> a(2);
  a[0] = 10;
  a[1] = 30;
  vector<int> b(2);
  b[0] = 15;
  b[1] = 20;
  
  double expected_logp = hypergeometric_log(n, N, a[0], b[0]) + hypergeometric_log(n, N, a[1], b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));  
}
TEST(ProbDistributionsHypergeometric,ivii) {
  int n = 5;
  vector<int> N(2);
  N[0] = 15;
  N[1] = 25;
  int a = 10;
  int b = 30;
  
  double expected_logp = hypergeometric_log(n, N[0], a, b) + hypergeometric_log(n, N[1], a, b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,iviv) {
  int n = 5;
  vector<int> N(2);
  N[0] = 15;
  N[1] = 12;
  int a = 20;
  vector<int> b(2);
  b[0] = 12;
  b[1] = 14;
  
  double expected_logp = hypergeometric_log(n, N[0], a, b[0]) + hypergeometric_log(n, N[1], a, b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,ivvi) {
  int n = 5;
  vector<int> N(2);
  N[0] = 12;
  N[1] = 20;
  vector<int> a(2);
  a[0] = 5;
  a[1] = 30;
  int b = 15;
  
  double expected_logp = hypergeometric_log(n, N[0], a[0], b) + hypergeometric_log(n, N[1], a[1], b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,ivvv) {
  int n = 5;
  vector<int> N(2);
  N[0] = 15;
  N[1] = 30;
  vector<int> a(2);
  a[0] = 6;
  a[1] = 12;
  vector<int> b(2);
  b[0] = 15;
  b[1] = 30;
  
  double expected_logp = hypergeometric_log(n, N[0], a[0], b[0]) + hypergeometric_log(n, N[1], a[1], b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,viii) {
  vector<int> n(2);
  n[0] = 5;
  n[1] = 3;
  int N = 15;
  int a = 10;
  int b = 20;

  double expected_logp = hypergeometric_log(n[0], N, a, b) + hypergeometric_log(n[1], N, a, b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,viiv) {
  vector<int> n(2);
  n[0] = 2;
  n[1] = 5;
  int N = 14;
  int a = 10;
  vector<int> b(2);
  b[0] = 12;
  b[1] = 15;
  
  double expected_logp = hypergeometric_log(n[0], N, a, b[0]) + hypergeometric_log(n[1], N, a, b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vivi) {
  vector<int> n(2);
  n[0] = 5;
  n[1] = 2;
  int N = 15;
  vector<int> a(2);
  a[0] = 10;
  a[1] = 20;
  int b = 30;
  
  double expected_logp = hypergeometric_log(n[0], N, a[0], b) + hypergeometric_log(n[1], N, a[1], b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vivv) {
  vector<int> n(2);
  n[0] = 4;
  n[1] = 6;
  int N = 15;
  vector<int> a(2);
  a[0] = 5;
  a[1] = 20;
  vector<int> b(2);
  b[0] = 15;
  b[1] = 10;
  
  double expected_logp = hypergeometric_log(n[0], N, a[0], b[0]) + hypergeometric_log(n[1], N, a[1], b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vvii) {
  vector<int> n(2);
  n[0] = 3;
  n[1] = 2;
  vector<int> N(2);
  N[0] = 10;
  N[1] = 8;
  int a = 5;
  int b = 30;
  
  double expected_logp = hypergeometric_log(n[0], N[0], a, b) + hypergeometric_log(n[1], N[1], a, b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vviv) {
  vector<int> n(2);
  n[0] = 4;
  n[1] = 1;
  vector<int> N(2);
  N[0] = 10;
  N[1] = 8;
  int a = 15;
  vector<int> b(2);
  b[0] = 20;
  b[1] = 16;
  
  double expected_logp = hypergeometric_log(n[0], N[0], a, b[0]) + hypergeometric_log(n[1], N[1], a, b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vvvi) {
  vector<int> n(2);
  n[0] = 3;
  n[1] = 4;
  vector<int> N(2);
  N[0] = 6;
  N[1] = 7;
  vector<int> a(2);
  a[0] = 10;
  a[1] = 12;
  int b = 20;
  
  double expected_logp = hypergeometric_log(n[0], N[0], a[0], b) + hypergeometric_log(n[1], N[1], a[1], b);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
}
TEST(ProbDistributionsHypergeometric,vvvv) {
  vector<int> n(2);
  n[0] = 5;
  n[1] = 0;
  vector<int> N(2);
  N[0] = 15;
  N[1] = 2;
  vector<int> a(2);
  a[0] = 10;
  a[1] = 3;
  vector<int> b(2);
  b[0] = 10;
  b[1] = 2;
  
  double expected_logp = hypergeometric_log(n[0], N[0], a[0], b[0]) + hypergeometric_log(n[1], N[1], a[1], b[1]);
  EXPECT_FLOAT_EQ(expected_logp, hypergeometric_log(n, N, a, b));
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

using stan::prob::hypergeometric_log;

TEST(ProbDistributionsHypergeometric,DefaultPolicy) {
  int n = 2;
  int N = 10;
  int a = 5;
  int b = 11;
    
  EXPECT_NO_THROW(hypergeometric_log(n,N,a,b));

  EXPECT_THROW(hypergeometric_log(6,N,a,b), std::domain_error) << "n > a";
  EXPECT_THROW(hypergeometric_log(n,N,a,7), std::domain_error) << "N-n > b";
  EXPECT_THROW(hypergeometric_log(n,17,a,b), std::domain_error) << "N > a+b";
}
TEST(ProbDistributionsHypergeometric,ErrnoPolicy) {
  int n = 2;
  int N = 10;
  int a = 5;
  int b = 11;
  double result;
  
  result = hypergeometric_log(n,N,a,b, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = hypergeometric_log(6,N,a,b, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "n > a";
  result = hypergeometric_log(n,N,a,7, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "N-n > b";
  result = hypergeometric_log(n,17,a,b, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "N > a+b";
}
