#include <stan/math/prim/mat/fun/eigenvalues_sym.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core/operator_addition.hpp>
#include <stan/math/rev/core/operator_divide_equal.hpp>
#include <stan/math/rev/core/operator_division.hpp>
#include <stan/math/rev/core/operator_equal.hpp>
#include <stan/math/rev/core/operator_greater_than.hpp>
#include <stan/math/rev/core/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/core/operator_less_than.hpp>
#include <stan/math/rev/core/operator_less_than_or_equal.hpp>
#include <stan/math/rev/core/operator_minus_equal.hpp>
#include <stan/math/rev/core/operator_multiplication.hpp>
#include <stan/math/rev/core/operator_multiply_equal.hpp>
#include <stan/math/rev/core/operator_not_equal.hpp>
#include <stan/math/rev/core/operator_plus_equal.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>
#include <stan/math/rev/core/operator_unary_decrement.hpp>
#include <stan/math/rev/core/operator_unary_increment.hpp>
#include <stan/math/rev/core/operator_unary_negative.hpp>
#include <stan/math/rev/core/operator_unary_not.hpp>
#include <stan/math/rev/core/operator_unary_plus.hpp>
#include <stan/math/fwd/core/operator_addition.hpp>
#include <stan/math/fwd/core/operator_division.hpp>
#include <stan/math/fwd/core/operator_equal.hpp>
#include <stan/math/fwd/core/operator_greater_than.hpp>
#include <stan/math/fwd/core/operator_greater_than_or_equal.hpp>
#include <stan/math/fwd/core/operator_less_than.hpp>
#include <stan/math/fwd/core/operator_less_than_or_equal.hpp>
#include <stan/math/fwd/core/operator_multiplication.hpp>
#include <stan/math/fwd/core/operator_not_equal.hpp>
#include <stan/math/fwd/core/operator_subtraction.hpp>
#include <stan/math/fwd/core/operator_unary_minus.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_fv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, exceptions_matrix_ffv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

TEST(AgradFwdMatrixEigenvaluesSym, matrix_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_fd res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_);
  EXPECT_FLOAT_EQ(3, res0(1).val_);
  EXPECT_FLOAT_EQ(0, res0(0).d_);
  EXPECT_FLOAT_EQ(2, res0(1).d_);
}
TEST(AgradFwdMatrixEigenvaluesSym, matrix_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_ffd res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val_);
  EXPECT_FLOAT_EQ(3, res0(1).val_.val_);
  EXPECT_FLOAT_EQ(0, res0(0).d_.val_);
  EXPECT_FLOAT_EQ(2, res0(1).d_.val_);
}

TEST(AgradFwdMatrixEigenvaluesSym, matrix_fv_1st_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_fv res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val());
  EXPECT_FLOAT_EQ(3, res0(1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0).d_.val());
  EXPECT_FLOAT_EQ(2, res0(1).d_.val());

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(1).val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.5,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(1.0,h[2]);
  EXPECT_FLOAT_EQ(0.5,h[3]);
}
TEST(AgradFwdMatrixEigenvaluesSym, matrix_fv_2nd_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_fv res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val());
  EXPECT_FLOAT_EQ(3, res0(1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0).d_.val());
  EXPECT_FLOAT_EQ(2, res0(1).d_.val());

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(1).d_.grad(z,h);
  EXPECT_NEAR(-1.110223e-16,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(-2.7755576e-17,h[2],1e-8);
  EXPECT_NEAR(1.110223e-16,h[3],1e-8);
}


TEST(AgradFwdMatrixEigenvaluesSym, matrix_ffv_1st_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_ffv res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3, res0(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0).d_.val_.val());
  EXPECT_FLOAT_EQ(2, res0(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(1).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.5,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(1.0,h[2]);
  EXPECT_FLOAT_EQ(0.5,h[3]);
}
TEST(AgradFwdMatrixEigenvaluesSym, matrix_ffv_2nd_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  stan::agrad::vector_ffv res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3, res0(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0).d_.val_.val());
  EXPECT_FLOAT_EQ(2, res0(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(1).d_.val_.grad(z,h);
  EXPECT_NEAR(-1.110223e-16,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(-2.7755576e-17,h[2],1e-8);
  EXPECT_NEAR(1.110223e-16,h[3],1e-8);
}
TEST(AgradFwdMatrixEigenvaluesSym, matrix_ffv_3rd_deriv) {
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

  stan::agrad::vector_ffv res0 = stan::math::eigenvalues_sym(m1);

  EXPECT_FLOAT_EQ(-1, res0(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3, res0(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0).d_.val_.val());
  EXPECT_FLOAT_EQ(2, res0(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(1).d_.d_.grad(z,h);
  EXPECT_NEAR(1.110223e-16,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(2.7755576e-17,h[2],1e-8);
  EXPECT_NEAR(-1.110223e-16,h[3],1e-8);
}

