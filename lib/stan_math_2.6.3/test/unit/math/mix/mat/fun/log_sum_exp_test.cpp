#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/log_sum_exp.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/log_sum_exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;
using stan::math::var;
using stan::math::log_sum_exp;
using stan::math::log_sum_exp;

TEST(AgradMixMatrixLogSumExp,vector_fv_1st_deriv) {
  using stan::math::vector_fv;

  vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,row_vector_fv_1st_deriv) {
  using stan::math::row_vector_fv;

  row_vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,matrix_fv_1st_deriv) {
  using stan::math::matrix_fv;

  matrix_fv b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 1.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val());

  std::vector<var> z;
  z.push_back(b(0,0).val_);
  z.push_back(b(0,1).val_);
  z.push_back(b(1,0).val_);
  z.push_back(b(1,1).val_);

  VEC h;
  a.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,vector_fv_2nd_deriv) {
  using stan::math::vector_fv;

  vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val());
  EXPECT_FLOAT_EQ(1.0320586,a.d_.val());

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,row_vector_fv_2nd_deriv) {
  using stan::math::row_vector_fv;

  row_vector_fv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_);
  z.push_back(b(1).val_);
  z.push_back(b(2).val_);
  z.push_back(b(3).val_);

  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,matrix_fv_2nd_deriv) {
  using stan::math::matrix_fv;

  matrix_fv b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 2.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<var> a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0,0).val_);
  z.push_back(b(0,1).val_);
  z.push_back(b(1,0).val_);
  z.push_back(b(1,1).val_);

  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,vector_ffv_1st_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,row_vector_ffv_1st_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,matrix_ffv_1st_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 1.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_.val());
  EXPECT_FLOAT_EQ(1,a.d_.val_.val());

  std::vector<var> z;
  z.push_back(b(0,0).val_.val_);
  z.push_back(b(0,1).val_.val_);
  z.push_back(b(1,0).val_.val_);
  z.push_back(b(1,1).val_.val_);

  VEC h;
  a.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.032058604,h[0]);
  EXPECT_FLOAT_EQ(0.087144315,h[1]);
  EXPECT_FLOAT_EQ(0.23688282,h[2]);
  EXPECT_FLOAT_EQ(0.64391428,h[3]);
}

TEST(AgradMixMatrixLogSumExp,vector_ffv_2nd_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,row_vector_ffv_2nd_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,matrix_ffv_2nd_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 2.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0,0).val_.val_);
  z.push_back(b(0,1).val_.val_);
  z.push_back(b(1,0).val_.val_);
  z.push_back(b(1,1).val_.val_);

  VEC h;
  a.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.031030849,h[0]);
  EXPECT_FLOAT_EQ(-0.0027937251,h[1]);
  EXPECT_FLOAT_EQ(-0.0075941323,h[2]);
  EXPECT_FLOAT_EQ(-0.020642992,h[3]);
}

TEST(AgradMixMatrixLogSumExp,vector_ffv_3rd_deriv) {
  using stan::math::vector_ffv;

  vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.029041238,h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992,h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178,h[2]);
  EXPECT_FLOAT_EQ(-0.019319421,h[3]);
}

TEST(AgradMixMatrixLogSumExp,row_vector_ffv_3rd_deriv) {
  using stan::math::row_vector_ffv;

  row_vector_ffv b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 2.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0).val_.val_);
  z.push_back(b(1).val_.val_);
  z.push_back(b(2).val_.val_);
  z.push_back(b(3).val_.val_);

  VEC h;
  a.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.029041238,h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992,h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178,h[2]);
  EXPECT_FLOAT_EQ(-0.019319421,h[3]);
}

TEST(AgradMixMatrixLogSumExp,matrix_ffv_3rd_deriv) {
  using stan::math::matrix_ffv;

  matrix_ffv b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 2.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;
  b(0).val_.d_ = 2.0;
  b(1).val_.d_ = 1.0;
  b(2).val_.d_ = 1.0;
  b(3).val_.d_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(b);

  std::vector<var> z;
  z.push_back(b(0,0).val_.val_);
  z.push_back(b(0,1).val_.val_);
  z.push_back(b(1,0).val_.val_);
  z.push_back(b(1,1).val_.val_);

  VEC h;
  a.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.029041238,h[0]);
  EXPECT_FLOAT_EQ(-0.0026145992,h[1]);
  EXPECT_FLOAT_EQ(-0.0071072178,h[2]);
  EXPECT_FLOAT_EQ(-0.019319421,h[3]);
}
