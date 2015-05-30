#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/wishart_log.hpp>
#include <stan/math/prim/mat/prob/wishart_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/variance.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;


TEST(ProbDistributionsWishart,LowerTriangular) {
  //Tests if only of the lower triangular portion of
  //outcome and scale matrices is taken
  using Eigen::MatrixXd;
  using stan::math::wishart_log;
  
  MatrixXd Sigma(4,4);
  MatrixXd Sigma_sym(4,4);
  MatrixXd Y(4,4);
  MatrixXd Y_sym(4,4);

  Y << 7.988168,  -10.955605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -15.454186,
    -14.474830,  49.215769,  60.08987, -20.481079,
    4.395895, -18.454186, -21.48108, 7.885833;

  Y_sym << 7.988168,  -9.555605, -14.474830,   4.395895,
    -9.555605,  44.750570,  49.215769, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.48108,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;
  
  Sigma_sym << 2.9983662,  0.2898776, -2.6505229,  0.1055911,
    0.2898776, 11.4803610,  7.1579931, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.586685,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  unsigned int dof = 5;
   
  EXPECT_EQ(wishart_log(Y,dof,Sigma), wishart_log(Y_sym,dof,Sigma));
  EXPECT_EQ(wishart_log(Y,dof,Sigma), wishart_log(Y,dof,Sigma_sym));
  EXPECT_EQ(wishart_log(Y,dof,Sigma), wishart_log(Y_sym,dof,Sigma_sym));
}
TEST(ProbDistributionsWishart,2x2) {
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  unsigned int dof = 3;
  
  double lp = log(8.658e-07); // computed with MCMCpack in R
 
  EXPECT_NEAR(lp, stan::math::wishart_log(Y,dof,Sigma), 0.01);
}
TEST(ProbDistributionsWishart,4x4) {
  Matrix<double,Dynamic,Dynamic> Y(4,4);
  Y << 7.988168,  -9.555605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.481079,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Matrix<double,Dynamic,Dynamic> Sigma(4,4);
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  double dof = 4;
  double log_p = log(8.034197e-10);
  EXPECT_NEAR(log_p, stan::math::wishart_log(Y,dof,Sigma),0.01);
  
  dof = 5;
  log_p = log(1.517951e-10);
  EXPECT_NEAR(log_p, stan::math::wishart_log(Y,dof,Sigma),0.01);
}
TEST(ProbDistributionsWishart,2x2Propto) {
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  unsigned int dof = 3;
 
  EXPECT_FLOAT_EQ(0.0, stan::math::wishart_log<true>(Y,dof,Sigma));
}
TEST(ProbDistributionsWishart,4x4Propto) {
  Matrix<double,Dynamic,Dynamic> Y(4,4);
  Y << 7.988168,  -9.555605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.481079,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Matrix<double,Dynamic,Dynamic> Sigma(4,4);
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  double dof = 4;
  EXPECT_FLOAT_EQ(0.0, stan::math::wishart_log<true>(Y,dof,Sigma));
  
  dof = 5;
  EXPECT_FLOAT_EQ(0.0, stan::math::wishart_log<true>(Y,dof,Sigma));
}

using stan::math::wishart_log;

TEST(ProbDistributionsWishart, error) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  Sigma.setIdentity();
  Y.setIdentity();
  nu = 1;
  EXPECT_NO_THROW(wishart_log(Y, nu, Sigma));
  
  nu = 5;
  Sigma.resize(2,1);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::invalid_argument);

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::invalid_argument);
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::invalid_argument);

  Sigma.resize(3,3);
  Sigma.setIdentity();
  Y.resize(3,3);
  Y.setIdentity();
  nu = 3;
  EXPECT_NO_THROW(wishart_log(Y, nu, Sigma));
  nu = 2;
  EXPECT_THROW(wishart_log(Y, nu, Sigma), std::domain_error);
}

TEST(ProbDistributionsWishart, error_check) {
  boost::random::mt19937 rng;

  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    2.0, 1.0, 3.0;
  EXPECT_NO_THROW(stan::math::wishart_rng(3.0, sigma,rng));

  EXPECT_THROW(stan::math::wishart_rng(-3.0, sigma,rng),std::domain_error);

  Matrix<double,Dynamic,Dynamic> sigma2(3,4);
  EXPECT_THROW(stan::math::wishart_rng(3.0, sigma2,rng),std::invalid_argument);
}

TEST(ProbDistributionsWishart, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 2.0,
    -3.0,  4.0, 0.0,
    2.0, 0.0, 3.0;
  int N = 10000;
  boost::math::chi_squared mydist(1);

  int count = 0;
  double avg = 0;
  double expect = sigma.rows() * std::log(2.0) + std::log(stan::math::determinant(sigma)) + boost::math::digamma(5.0 / 2.0) + boost::math::digamma(4.0 / 2.0) + boost::math::digamma(3.0 / 2.0);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> a(sigma.rows(),sigma.rows());
  while (count < N) {
    a = stan::math::wishart_rng(5.0, sigma, rng);
    avg += std::log(stan::math::determinant(a)) / N;
    count++;
   }
 
  double chi = (expect - avg) * (expect - avg) / expect;

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}


TEST(ProbDistributionsWishart, SpecialRNGTest) {
  //For any vector C != 0
  //(C' * W * C) / (C' * S * C)
  //must be chi-square distributed with df = k
  //which has mean = k and variance = 2k
  
  boost::random::mt19937 rng(1234);
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd sigma(3,3);
  MatrixXd sigma_sym(3,3);
  
  //wishart_rng should take only the lower part
  sigma << 9.0, -3.0, 1.0,
    2.0,  4.0, -1.0,
    2.0, 1.0, 3.0;

  sigma_sym << 9.0, 2.0, 2.0,
    2.0,  4.0, 1.0,
    2.0, 1.0, 3.0;
  
  VectorXd C(3);
  C << 2, 1, 3;
  
  size_t N = 1e4;
  int k = 20;
  double tol = 0.2; //tolerance for variance
  std::vector<double> acum;
  acum.reserve(N);
  for (size_t i = 0; i < N; i++)
    acum.push_back((C.transpose() * stan::math::wishart_rng(k, sigma, rng) * C)(0) /
           (C.transpose() * sigma_sym * C)(0));
  

  EXPECT_NEAR(1, stan::math::mean(acum) / k, std::pow(tol, 2));
  EXPECT_NEAR(1, stan::math::variance(acum) / (2 * k), tol);
}
