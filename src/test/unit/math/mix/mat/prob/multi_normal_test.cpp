#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_normal_log.hpp>
#include <stan/math/prim/mat/prob/multi_normal_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/rev/scal/fun/is_nan.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormal,fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Matrix<fvar<var>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<var>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  fvar<var> res = stan::math::multi_normal_log(y,mu,Sigma);
  EXPECT_FLOAT_EQ(-11.73908, res.val_.val());
  EXPECT_FLOAT_EQ(0.54899865, res.d_.val());
}

TEST(ProbDistributionsMultiNormal,fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Matrix<fvar<fvar<var> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  fvar<fvar<var> > res = stan::math::multi_normal_log(y,mu,Sigma);
  EXPECT_FLOAT_EQ(-11.73908, res.val_.val_.val());
  EXPECT_FLOAT_EQ(0.54899865, res.d_.val_.val());
}
