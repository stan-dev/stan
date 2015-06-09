#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/lkj_cov_log.hpp>

TEST(ProbDistributionsLkjCorr,testIdentity) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  srand(time(0));
  double eta = rand() / double(RAND_MAX) + 0.5;
  double f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::math::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::math::lkj_corr_log(Sigma, eta));
}


TEST(ProbDistributionsLkjCorr,testHalf) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setConstant(0.5);
  Sigma.diagonal().setOnes();
  double eta = rand() / double(RAND_MAX) + 0.5;
  double f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f + (eta - 1.0) * log(0.3125), stan::math::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::math::lkj_corr_log(Sigma, eta));
}

TEST(ProbDistributionsLkjCorr,Sigma) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = rand() / double(RAND_MAX) + 0.5;
  EXPECT_NO_THROW (stan::math::lkj_corr_log(Sigma, eta));
  
  EXPECT_THROW (stan::math::lkj_corr_log(Sigma, -eta), std::domain_error);
  
  Sigma = Sigma * -1.0;
  EXPECT_THROW (stan::math::lkj_corr_log(Sigma, eta), std::domain_error);
  Sigma = Sigma * (0.0 / 0.0);
  EXPECT_THROW (stan::math::lkj_corr_log(Sigma, eta), std::domain_error);
}
