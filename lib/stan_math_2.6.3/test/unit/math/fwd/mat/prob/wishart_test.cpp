#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/wishart_log.hpp>
#include <stan/math/prim/mat/prob/wishart_rng.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_ldlt.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/variance.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsWishart, fvar_double) {
  using stan::math::fvar;
  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<fvar<double>,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  for (int i = 0; i < 4; i++) {
    Sigma(i).d_ = 1.0;
    Y(i).d_ = 1.0;
  }

  unsigned int dof = 3;
  
  double lp = log(8.658e-07); // computed with MCMCpack in R
 
  EXPECT_NEAR(lp, stan::math::wishart_log(Y,dof,Sigma).val_, 0.01);
  EXPECT_NEAR(-0.76893887, stan::math::wishart_log(Y,dof,Sigma).d_, 0.01);
}

TEST(ProbDistributionsWishart, fvar_fvar_double) {
  using stan::math::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  for (int i = 0; i < 4; i++) {
    Sigma(i).d_.val_ = 1.0;
    Y(i).d_.val_ = 1.0;
  }

  unsigned int dof = 3;
  
  double lp = log(8.658e-07); // computed with MCMCpack in R
 
  EXPECT_NEAR(lp, stan::math::wishart_log(Y,dof,Sigma).val_.val_, 0.01);
  EXPECT_NEAR(-0.76893887, stan::math::wishart_log(Y,dof,Sigma).d_.val_, 0.01);
}
