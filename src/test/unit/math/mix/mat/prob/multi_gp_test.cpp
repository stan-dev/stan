#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_normal_log.hpp>
#include <stan/math/prim/mat/prob/multi_gp_log.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
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

TEST(ProbDistributionsMultiGP,fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  Matrix<fvar<var>,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<var>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<var>,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_ = 1.0;
    if (i < 3)
      w(i).d_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3)
        y(i,j).d_ = 1.0;
    }
  }

  fvar<var> lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<var>,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<var>,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::math::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val(), stan::math::multi_gp_log(y,Sigma,w).val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::math::multi_gp_log(y,Sigma,w).d_.val());
}

TEST(ProbDistributionsMultiGP,fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_.val_ = 1.0;
    if (i < 3)
      w(i).d_.val_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3)
        y(i,j).d_.val_ = 1.0;
    }
  }

  fvar<fvar<var> > lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<fvar<var> >,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<fvar<var> >,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::math::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val_.val(), stan::math::multi_gp_log(y,Sigma,w).val_.val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::math::multi_gp_log(y,Sigma,w).d_.val_.val());
}
