#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left.hpp>
#include <stan/math/fwd/mat/fun/mdivide_right.hpp>

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>

#include <stan/math/prim/mat/prob/inv_wishart_log.hpp>
#include <stan/math/prim/mat/prob/inv_wishart_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using stan::math::inv_wishart_log;


TEST(ProbDistributionsInvWishart,fvar_double) {
  using stan::math::fvar;

  Matrix<fvar<double>,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::math::inv_wishart_log(Y,dof,Sigma).val_, 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::math::inv_wishart_log(Y,dof,Sigma).d_, 0.01);
}

TEST(ProbDistributionsInvWishart,fvar_fvar_double) {
  using stan::math::fvar;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::math::inv_wishart_log(Y,dof,Sigma).val_.val_, 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::math::inv_wishart_log(Y,dof,Sigma).d_.val_, 0.01);
}

