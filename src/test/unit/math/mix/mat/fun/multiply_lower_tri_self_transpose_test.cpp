#include <stan/math/fwd/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

using stan::math::fvar;
TEST(AgradMixMatrixMultiplyLowerTriSelfTranspose, fv_3x3_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fv output = stan::math::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixMultiplyLowerTriSelfTranspose, fv_3x3_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::math::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixMultiplyLowerTriSelfTranspose, fv_3x2_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(3,2);
  Z << 1, 0, 0,
    2, 3, 0;
  matrix_fv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fv output = stan::math::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 2,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 2,output(1,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,2).d_.val());
  EXPECT_FLOAT_EQ( 8,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixMultiplyLowerTriSelfTranspose, fv_3x2_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  matrix_fv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::math::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
