#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>

TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_fd) {
  stan::math::matrix_fd m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

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
  stan::math::matrix_ffd m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

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
  stan::math::matrix_fd m0(2,2);
  m0 << 1, 2, 2, 4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::math::matrix_fd res = cholesky_decompose(m0);

  EXPECT_FLOAT_EQ(1, res(0,0).val_);
  EXPECT_FLOAT_EQ(0, res(0,1).val_);
  EXPECT_FLOAT_EQ(2, res(1,0).val_);
  EXPECT_FLOAT_EQ(4, res(1,1).val_);
  EXPECT_FLOAT_EQ(0.5, res(0,0).d_);
  EXPECT_FLOAT_EQ(0, res(0,1).d_);
  EXPECT_FLOAT_EQ(0, res(1,0).d_);
  EXPECT_FLOAT_EQ(1, res(1,1).d_);
}
TEST(AgradFwdMatrixCholeskyDecompose, mat_ffd) {
  stan::math::matrix_ffd m0(2,2);
  m0 << 1, 2, 2, 4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::math::matrix_ffd res = cholesky_decompose(m0);

  EXPECT_FLOAT_EQ(1, res(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0, res(0,1).val_.val_);
  EXPECT_FLOAT_EQ(2, res(1,0).val_.val_);
  EXPECT_FLOAT_EQ(4, res(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.5, res(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val_);
  EXPECT_FLOAT_EQ(1, res(1,1).d_.val_);
}
