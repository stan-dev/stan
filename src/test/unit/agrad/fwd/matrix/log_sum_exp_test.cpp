#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/log_sum_exp.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/log_sum_exp.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
using stan::agrad::var;
using stan::agrad::log_sum_exp;
using stan::math::log_sum_exp;

TEST(AgradFwdMatrixLogSumExp,vector_fd) {
  using stan::agrad::vector_fd;

  vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_);
  EXPECT_FLOAT_EQ(1,a.d_);
}
TEST(AgradFwdMatrixLogSumExp,row_vector_fd) {
  using stan::agrad::row_vector_fd;

  row_vector_fd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_);
  EXPECT_FLOAT_EQ(1,a.d_);
}

TEST(AgradFwdMatrixLogSumExp,matrix_fd) {
  using stan::agrad::matrix_fd;

  matrix_fd b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 1.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<double> a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_);
  EXPECT_FLOAT_EQ(1,a.d_);
}

TEST(AgradFwdMatrixLogSumExp,vector_ffd) {
  using stan::agrad::vector_ffd;

  vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_);
  EXPECT_FLOAT_EQ(1,a.d_.val_);
}
TEST(AgradFwdMatrixLogSumExp,row_vector_ffd) {
  using stan::agrad::row_vector_ffd;

  row_vector_ffd b(4);
  b << 1, 2, 3, 4;
  b(0).d_ = 1.0;
  b(1).d_ = 1.0;
  b(2).d_ = 1.0;
  b(3).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_);
  EXPECT_FLOAT_EQ(1,a.d_.val_);
}

TEST(AgradFwdMatrixLogSumExp,matrix_ffd) {
  using stan::agrad::matrix_ffd;

  matrix_ffd b(2,2);
  b << 1, 2, 3, 4;
  b(0,0).d_ = 1.0;
  b(0,1).d_ = 1.0;
  b(1,0).d_ = 1.0;
  b(1,1).d_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(b);

  EXPECT_FLOAT_EQ(4.4401898,a.val_.val_);
  EXPECT_FLOAT_EQ(1,a.d_.val_);
}

TEST(AgradFwdMatrixLogSumExp,vector_fv_1st_deriv) {
  using stan::agrad::vector_fv;

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

TEST(AgradFwdMatrixLogSumExp,row_vector_fv_1st_deriv) {
  using stan::agrad::row_vector_fv;

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

TEST(AgradFwdMatrixLogSumExp,matrix_fv_1st_deriv) {
  using stan::agrad::matrix_fv;

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

TEST(AgradFwdMatrixLogSumExp,vector_fv_2nd_deriv) {
  using stan::agrad::vector_fv;

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

TEST(AgradFwdMatrixLogSumExp,row_vector_fv_2nd_deriv) {
  using stan::agrad::row_vector_fv;

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

TEST(AgradFwdMatrixLogSumExp,matrix_fv_2nd_deriv) {
  using stan::agrad::matrix_fv;

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

TEST(AgradFwdMatrixLogSumExp,vector_ffv_1st_deriv) {
  using stan::agrad::vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,row_vector_ffv_1st_deriv) {
  using stan::agrad::row_vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,matrix_ffv_1st_deriv) {
  using stan::agrad::matrix_ffv;

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

TEST(AgradFwdMatrixLogSumExp,vector_ffv_2nd_deriv) {
  using stan::agrad::vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,row_vector_ffv_2nd_deriv) {
  using stan::agrad::row_vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,matrix_ffv_2nd_deriv) {
  using stan::agrad::matrix_ffv;

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

TEST(AgradFwdMatrixLogSumExp,vector_ffv_3rd_deriv) {
  using stan::agrad::vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,row_vector_ffv_3rd_deriv) {
  using stan::agrad::row_vector_ffv;

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

TEST(AgradFwdMatrixLogSumExp,matrix_ffv_3rd_deriv) {
  using stan::agrad::matrix_ffv;

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
