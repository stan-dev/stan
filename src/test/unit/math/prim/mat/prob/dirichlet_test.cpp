#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/dirichlet_log.hpp>
#include <stan/math/prim/mat/prob/dirichlet_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::VectorXd;

TEST(ProbDistributions,Dirichlet) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.6931472, stan::math::dirichlet_log(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(-43.40045, stan::math::dirichlet_log(theta2,alpha2));
}

TEST(ProbDistributions,DirichletPropto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.0, stan::math::dirichlet_log<true>(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(0.0, stan::math::dirichlet_log<true>(theta2,alpha2));
}

TEST(ProbDistributions,DirichletBounds) {
  Matrix<double,Dynamic,1> good_alpha(2,1), bad_alpha(2,1);
  Matrix<double,Dynamic,1> good_theta(2,1), bad_theta(2,1);

  good_theta << 0.25, 0.75;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(stan::math::dirichlet_log(good_theta,good_alpha));

  good_theta << 1.0, 0.0;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(stan::math::dirichlet_log(good_theta,good_alpha))
    << "elements of theta can be 0";


  bad_theta << 0.25, 0.25;
  EXPECT_THROW(stan::math::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "sum of theta is not 1";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(stan::math::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "theta has element less than 0";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(stan::math::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "theta has element less than 0";

  bad_alpha << 0.0, 1.0;
  EXPECT_THROW(stan::math::dirichlet_log(good_theta,bad_alpha),
               std::domain_error)
    << "alpha has element equal to 0";

  bad_alpha << -0.5, 1.0;
  EXPECT_THROW(stan::math::dirichlet_log(good_theta,bad_alpha),
               std::domain_error)
    << "alpha has element less than 0";

  bad_alpha = Matrix<double,Dynamic,1>(4,1);
  bad_alpha << 1, 2, 3, 4;
  EXPECT_THROW(stan::math::dirichlet_log(good_theta,bad_alpha),
               std::invalid_argument)
    << "size mismatch: theta is a 2-vector, alpha is a 4-vector";
}

double chi_square(std::vector<int> bin, std::vector<double> expect) {
  double chi = 0;
  for (size_t j = 0; j < bin.size(); j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);
  return chi;
}

void test_dirichlet3_1(VectorXd alpha) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));

  // bins 0 vs. 1 + 2
  boost::math::beta_distribution<> dist(alpha(0), alpha(1) + alpha(2));
  boost::math::chi_squared mydist(K - 1);

  std::vector<double> loc(K - 1);
  for (int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i / static_cast<double>(K));

  std::vector<int> bin(K, 0);
  std::vector<double> expect(K, N / static_cast<double>(K));

  for (int count = 0; count < N; ++count) {
    Eigen::VectorXd theta = stan::math::dirichlet_rng(alpha,rng);
    int i;
    for (i = 0; i < K-1 && theta(0) > loc[i]; ++i) ;
    ++bin[i];
  }
  EXPECT_TRUE(chi_square(bin,expect) < quantile(complement(mydist, 1e-6)));  
}

void test_dirichlet3_2(VectorXd alpha) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<> dist(alpha(1), alpha(0) + alpha(2));
  boost::math::chi_squared mydist(K - 1);

  std::vector<double> loc(K - 1);
  for(size_t i = 0; i < loc.size(); i++)
    loc[i] = quantile(dist, (i + 1.0) / K);


  std::vector<int> bin(K, 0);
  std::vector<double> expect(K);
  for (int i = 0 ; i < K; i++)
    expect[i] = N / K;

  for (int count = 0; count < N; ++count) {
    VectorXd a = stan::math::dirichlet_rng(alpha,rng);
    int i = 0;
    while (i < K-1 && a(1) > loc[i]) 
      ++i;
    ++bin[i];
   }

  EXPECT_TRUE(chi_square(bin, expect) < quantile(complement(mydist, 1e-6)));
}



TEST(ProbDistributionsDirichlet, rngTest) {
  VectorXd alpha(3);
  alpha << 2.0, 3.0, 11.0;
  test_dirichlet3_1(alpha);
  test_dirichlet3_2(alpha);

  VectorXd beta(3);
  beta << 0.1, 0.01, 0.2;
  test_dirichlet3_1(beta);
  test_dirichlet3_2(beta);
}

TEST(ProbDistributionsDirichlet, random) {
  boost::random::mt19937 rng;
  VectorXd alpha(3);
  alpha << 2.0, 3.0, 11.0;
  EXPECT_NO_THROW(stan::math::dirichlet_rng(alpha, rng));

  VectorXd beta(3);
  beta << 0.001, 0.0001, 1e-10;
  EXPECT_NO_THROW(stan::math::dirichlet_rng(beta, rng));
}
