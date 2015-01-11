#include <stan/math/matrix/cholesky_decompose.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>


TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_fd) {
  stan::agrad::matrix_fd m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

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
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}
TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_fv) {
  stan::agrad::matrix_fv m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}
TEST(AgradFwdMatrixCholeskyDecompose, exception_mat_ffv) {
  stan::agrad::matrix_ffv m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_fd) {
  stan::agrad::matrix_fd m0(2,2);
  m0 << 1, 2, 2, 4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_fd res = cholesky_decompose(m0);

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
  stan::agrad::matrix_ffd m0(2,2);
  m0 << 1, 2, 2, 4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_ffd res = cholesky_decompose(m0);

  EXPECT_FLOAT_EQ(1, res(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0, res(0,1).val_.val_);
  EXPECT_FLOAT_EQ(2, res(1,0).val_.val_);
  EXPECT_FLOAT_EQ(4, res(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.5, res(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val_);
  EXPECT_FLOAT_EQ(1, res(1,1).d_.val_);
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_fv_1st_deriv) {
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2, 4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_fv res = cholesky_decompose(m1);

  EXPECT_FLOAT_EQ(1, res(0,0).val_.val());
  EXPECT_FLOAT_EQ(0, res(0,1).val_.val());
  EXPECT_FLOAT_EQ(2, res(1,0).val_.val());
  EXPECT_FLOAT_EQ(4, res(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.5, res(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, res(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val());
  EXPECT_FLOAT_EQ(1, res(1,1).d_.val());

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.5,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_fv_2nd_deriv) {
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2, 4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_fv res = cholesky_decompose(m1);

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_ffv_1st_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2, 4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_ffv res = cholesky_decompose(m1);

  EXPECT_FLOAT_EQ(1, res(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(2, res(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(4, res(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.5, res(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1, res(1,1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0,0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.5,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixCholeskyDecompose, mat_ffv_2nd_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2, 4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_ffv res = cholesky_decompose(m1);

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0,0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixCholeskyDecompose, mat_ffv_3rd_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2, 4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;

  using stan::math::cholesky_decompose;

  stan::agrad::matrix_ffv res = cholesky_decompose(m1);

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.375,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
