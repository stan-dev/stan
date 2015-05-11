#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <cmath>

template <typename T>
void deriv_chol_fwd(const Eigen::Matrix<T, -1, -1>& parent_mat,
                    Eigen::Matrix<double,-1, -1>& vals,
                    Eigen::Matrix<double, -1, -1>& gradients) {
  using stan::math::value_of_rec;
  Eigen::Matrix<double,2,2> parent_mat_d = value_of_rec(parent_mat);
  vals(0,0) = sqrt(parent_mat_d(0,0));
  vals(1,0) = parent_mat_d(1,0) / sqrt(parent_mat_d(0,0));
  vals(0,1) = 0.0;
  vals(1,1) = sqrt(parent_mat_d(1,1) - pow(vals(1,0), 2));

  double pow_neg_half_00 = pow(parent_mat_d(0,0), -0.5);
  double pow_neg_half_comb = pow(parent_mat_d(1,1) 
                                 - pow(parent_mat_d(1,0),2.0) 
                                 / parent_mat_d(0,0), -0.5);
  gradients(0,0) = pow_neg_half_00 / 2 * value_of_rec(parent_mat(0,0).d_);
  gradients(1,0) = -0.5 * value_of_rec(parent_mat(0,0).d_) 
    * pow(parent_mat_d(0,0), -1.5)
    + value_of_rec(parent_mat(1,0).d_) * pow_neg_half_00 
    * parent_mat_d(1,0);
  gradients(0,1) = 0.0;
  gradients(1,1) = 0.5 * pow_neg_half_comb 
    * (value_of_rec(parent_mat(1,1).d_) 
       * - 2 * parent_mat_d(1,0) / parent_mat_d(0,0) 
       * value_of_rec(parent_mat(1,0).d_) 
       + pow(parent_mat_d(1,0),2.0) / pow(parent_mat_d(0,0),2.0) 
       * value_of_rec(parent_mat(0,0).d_));
}

TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_fd) {
  stan::agrad::matrix_fd m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m),std::domain_error);

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::invalid_argument);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}
TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_ffd) {
  stan::agrad::matrix_ffd m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m),std::domain_error);

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::invalid_argument);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}
TEST(AgradFwdMatrixCholeskyDecompose, mat_fd) {
  stan::agrad::matrix_fd m0(2,2);
  m0 << 2, 1, 1, 2;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_fd res = cholesky_decompose(m0);
  Eigen::Matrix<double,-1,-1> res_mat(2,2);
  Eigen::Matrix<double,-1,-1> d_mat(2,2);
  deriv_chol_fwd(m0, res_mat, d_mat);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < i; ++j) {
     EXPECT_FLOAT_EQ(res_mat(i,j), res(i,j).val_);
     EXPECT_FLOAT_EQ(d_mat(i,j),res(i,j).d_) << "Row: " << i
       << "Col: " << j;
    }
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_ffd) {
  stan::agrad::matrix_ffd m0(2,2);
  m0 << 4, 1, 1, 4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_ffd res = cholesky_decompose(m0);
  Eigen::Matrix<double,-1,-1> res_mat(2,2);
  Eigen::Matrix<double,-1,-1> d_mat(2,2);
  deriv_chol_fwd(m0, res_mat, d_mat);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < i; ++j) {
     EXPECT_FLOAT_EQ(res_mat(i,j), res(i,j).val_.val_);
     EXPECT_FLOAT_EQ(d_mat(i,j),res(i,j).d_.val_) << "Row: " << i
       << " Col: " << j;
    }
}
