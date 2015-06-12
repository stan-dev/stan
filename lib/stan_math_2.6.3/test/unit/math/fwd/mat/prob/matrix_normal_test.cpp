#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/matrix_normal_prec_log.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/mat/fun/trace_gen_quad_form.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMatrixNormal,fvar_double) {
  using stan::math::fvar;

  Matrix<fvar<double>,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<double>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<double>,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3) {
        mu(i,j).d_ = 1.0;
        y(i,j).d_ = 1.0;
        if (j < 3)
          D(i,j).d_ = 1.0;
      }
    } 

  fvar<double> lp_ref = stan::math::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_);
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_);
}

TEST(ProbDistributionsMatrixNormal,fvar_fvar_double) {
  using stan::math::fvar;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3) {
        mu(i,j).d_.val_ = 1.0;
        y(i,j).d_.val_ = 1.0;
        if (j < 3)
          D(i,j).d_.val_ = 1.0;
      }
    } 

  fvar<fvar<double> > lp_ref = stan::math::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_.val_);
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_.val_);
}

