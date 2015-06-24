#include <stan/math/prim/mat/fun/eigenvalues_sym.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_fd) {
  stan::math::matrix_fd m0;
  stan::math::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_ffd) {
  stan::math::matrix_ffd m0;
  stan::math::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, matrix_fd) {
  stan::math::matrix_fd m0;
  stan::math::matrix_fd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::vector_fd res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_);
  EXPECT_FLOAT_EQ(3, res0(1).val_);
  EXPECT_FLOAT_EQ(0, res0(0).d_);
  EXPECT_FLOAT_EQ(2, res0(1).d_);
}
TEST(AgradFwdMatrixEigenvaluesSym, matrix_ffd) {
  stan::math::matrix_ffd m0;
  stan::math::matrix_ffd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::math::vector_ffd res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val_);
  EXPECT_FLOAT_EQ(3, res0(1).val_.val_);
  EXPECT_FLOAT_EQ(0, res0(0).d_.val_);
  EXPECT_FLOAT_EQ(2, res0(1).d_.val_);
}
