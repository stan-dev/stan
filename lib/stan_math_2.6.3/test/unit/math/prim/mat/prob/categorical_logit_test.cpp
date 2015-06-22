#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <stan/math/prim/mat/prob/categorical_logit_log.hpp>
#include <stan/math/prim/mat/fun/log_softmax.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::log_softmax;
using stan::math::log_softmax;

TEST(ProbDistributionsCategoricalLogit,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  Matrix<double,Dynamic,1> theta_log_softmax = log_softmax(theta);

  EXPECT_FLOAT_EQ(theta_log_softmax[0], stan::math::categorical_logit_log(1,theta));
  EXPECT_FLOAT_EQ(theta_log_softmax[1], stan::math::categorical_logit_log(2,theta));
  EXPECT_FLOAT_EQ(theta_log_softmax[2], stan::math::categorical_logit_log(3,theta));
}

TEST(ProbDistributionsCategoricalLogit,CategoricalVectorized) {
  Matrix<double,Dynamic,1> theta(3);
  theta << -1, 2, -10;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::math::categorical_logit_log(ns,theta));

  Matrix<double,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0] + theta_log_softmax[1] + theta_log_softmax[0],
                  stan::math::categorical_logit_log(ms,theta));
}



TEST(ProbDistributionsCategoricalLogit,Propto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, 10;
  EXPECT_FLOAT_EQ(0, stan::math::categorical_logit_log<true>(1,theta));
  EXPECT_FLOAT_EQ(0, stan::math::categorical_logit_log<true>(3,theta));
}


TEST(ProbDistributionsCategoricalLogit, error) {
  using stan::math::categorical_logit_log;

  unsigned int n = 1;
  unsigned int N = 3;
  Matrix<double,Dynamic,1> theta(N,1);
  theta << 0.3, 0.5, 0.2;

  EXPECT_NO_THROW(categorical_logit_log(N, theta));
  EXPECT_NO_THROW(categorical_logit_log(n, theta));
  EXPECT_NO_THROW(categorical_logit_log(2, theta));
  EXPECT_THROW(categorical_logit_log(N+1, theta), std::domain_error);
  EXPECT_THROW(categorical_logit_log(0, theta), std::domain_error);

  theta(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(categorical_logit_log(1, theta), std::domain_error);

  theta(1) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(categorical_logit_log(1, theta), std::domain_error);

  std::vector<int> ns(2);
  ns[0] = 1;
  ns[1] = 2;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);

  theta << 0.3, 0.5, 0.2;
  EXPECT_NO_THROW(categorical_logit_log(ns, theta));
  
  ns[0] = -1;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);

  ns[0] = 1;
  ns[1] = 12;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);
}
