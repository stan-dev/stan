#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/eigenvectors_sym.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(AgradMixMatrixEigenvectorsSym, excepts_fv) {
  stan::math::matrix_fv m0;
  stan::math::matrix_fv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_fv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::invalid_argument);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::invalid_argument);
}
TEST(AgradMixMatrixEigenvectorsSym, excepts_ffv) {
  stan::math::matrix_ffv m0;
  stan::math::matrix_ffv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_ffv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::invalid_argument);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::invalid_argument);
}

TEST(AgradMixMatrixEigenvectorsSym, matrix_fv_1st_deriv) {
  stan::math::matrix_fv m0;
  stan::math::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_fv res0 = stan::math::eigenvectors_sym(m1);

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
TEST(AgradMixMatrixEigenvectorsSym, matrix_fv_2nd_deriv) {
  stan::math::matrix_fv m0;
  stan::math::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_fv res0 = stan::math::eigenvectors_sym(m1);

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

TEST(AgradMixMatrixEigenvectorsSym, matrix_ffv_1st_deriv) {
  stan::math::matrix_ffv m0;
  stan::math::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

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
TEST(AgradMixMatrixEigenvectorsSym, matrix_ffv_2nd_deriv) {
  stan::math::matrix_ffv m0;
  stan::math::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

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

TEST(AgradMixMatrixEigenvectorsSym, matrix_ffv_3rd_deriv) {
  stan::math::matrix_ffv m0;
  stan::math::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;

  stan::math::matrix_ffv res0 = stan::math::eigenvectors_sym(m1);

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
