#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/eigenvectors_sym.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdMatrixEigenvectorsSym, excepts_fd) {
  stan::math::matrix_fd m0;
  stan::math::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_fd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::invalid_argument);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::invalid_argument);
}
TEST(AgradFwdMatrixEigenvectorsSym, excepts_ffd) {
  stan::math::matrix_ffd m0;
  stan::math::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_ffd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::invalid_argument);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::invalid_argument);
}
TEST(AgradFwdMatrixEigenvectorsSym, matrix_fd) {
  stan::math::matrix_fd m0;
  stan::math::matrix_fd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_fd res0 = stan::math::eigenvectors_sym(m1);

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
  stan::math::matrix_ffd m0;
  stan::math::matrix_ffd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::matrix_ffd res0 = stan::math::eigenvectors_sym(m1);

  EXPECT_FLOAT_EQ(-0.70710677, res0(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(0,1).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_);
}
