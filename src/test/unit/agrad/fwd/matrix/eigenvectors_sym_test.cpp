#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/math/matrix/eigenvectors_sym.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixEigenvectorsSym, excepts_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_fd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::domain_error);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::domain_error);
}
TEST(AgradFwdMatrixEigenvectorsSym, excepts_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_ffd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::domain_error);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::domain_error);
}
TEST(AgradFwdMatrixEigenvectorsSym, excepts_fv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_fv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::domain_error);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::domain_error);
}
TEST(AgradFwdMatrixEigenvectorsSym, excepts_ffv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_ffv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::domain_error);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::domain_error);
}

TEST(AgradFwdMatrixEigenvectorsSym, matrix_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_fd res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_);
  EXPECT_FLOAT_EQ(0, res0(0,0).d_);
  EXPECT_FLOAT_EQ(0, res0(0,1).d_);
  EXPECT_FLOAT_EQ(0, res0(1,0).d_);
  EXPECT_FLOAT_EQ(0, res0(1,1).d_);
}

TEST(AgradFwdMatrixEigenvectorsSym, matrix_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_ffd res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_);
}

TEST(AgradFwdMatrixEigenvectorsSym, matrix_fv_1st_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_fv res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val());


  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.17677669,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.17677669,h[3]);
}
TEST(AgradFwdMatrixEigenvectorsSym, matrix_fv_2nd_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_fv res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val());


  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.088388346,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.088388346,h[3]);
}

TEST(AgradFwdMatrixEigenvectorsSym, matrix_ffv_1st_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.17677669,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-0.17677669,h[3]);
}
TEST(AgradFwdMatrixEigenvectorsSym, matrix_ffv_2nd_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.088388346,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.088388346,h[3]);
}

TEST(AgradFwdMatrixEigenvectorsSym, matrix_ffv_3rd_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;

  stan::agrad::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).d_.d_.grad(z,h);
  EXPECT_NEAR(0.088388346,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(-0.088388346,h[3],1e-8);
}
