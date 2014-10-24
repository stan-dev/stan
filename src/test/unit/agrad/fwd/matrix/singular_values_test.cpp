#include <stan/math/matrix/singular_values.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixSingularValues, mat_fd) {
  stan::agrad::matrix_fd m0(2,2);
  stan::agrad::vector_fd res;

  m0 << 1,2,3,4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m0);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_);
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_);
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_);
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_);
}

TEST(AgradFwdMatrixSingularValues, mat_ffd) {
  stan::agrad::matrix_ffd m0(2,2);
  stan::agrad::vector_ffd res;

  m0 << 1,2,3,4;
  m0(0,0).d_ = 1.0;
  m0(0,1).d_ = 1.0;
  m0(1,0).d_ = 1.0;
  m0(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m0);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val_);
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val_);
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val_);
}

TEST(AgradFwdMatrixSingularValues, mat_fv_1st_deriv) {
  stan::agrad::matrix_fv m1(2,2);
  stan::agrad::vector_fv res;

  m1 << 1,2,3,4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m1);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val());
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val());
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val());
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val());

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res(0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.23304246,h[0]);
  EXPECT_FLOAT_EQ(0.33068839,h[1]);
  EXPECT_FLOAT_EQ(0.52680451,h[2]);
  EXPECT_FLOAT_EQ(0.74753821,h[3]);
}
TEST(AgradFwdMatrixSingularValues, mat_fv_2nd_deriv) {
  stan::agrad::matrix_fv m1(2,2);
  stan::agrad::vector_fv res;

  m1 << 1,2,3,4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m1);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val());
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val());
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val());
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val());

  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res(0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.083204068,h[0]);
  EXPECT_FLOAT_EQ(0.083111323,h[1]);
  EXPECT_FLOAT_EQ(0.0076820431,h[2]);
  EXPECT_FLOAT_EQ(-0.068118215,h[3]);
}

TEST(AgradFwdMatrixSingularValues, mat_ffv_1st_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  stan::agrad::vector_ffv res;

  m1 << 1,2,3,4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m1);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.23304246,h[0]);
  EXPECT_FLOAT_EQ(0.33068839,h[1]);
  EXPECT_FLOAT_EQ(0.52680451,h[2]);
  EXPECT_FLOAT_EQ(0.74753821,h[3]);
}
TEST(AgradFwdMatrixSingularValues, mat_ffv_2nd_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  stan::agrad::vector_ffv res;

  m1 << 1,2,3,4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m1);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.083204068,h[0]);
  EXPECT_FLOAT_EQ(0.083111323,h[1]);
  EXPECT_FLOAT_EQ(0.0076820431,h[2]);
  EXPECT_FLOAT_EQ(-0.068118215,h[3]);
}

TEST(AgradFwdMatrixSingularValues, mat_ffv_3rd_deriv) {
  stan::agrad::matrix_ffv m1(2,2);
  stan::agrad::vector_ffv res;

  m1 << 1,2,3,4;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;

  using stan::math::singular_values;

  res = singular_values(m1);
  EXPECT_FLOAT_EQ(5.4649858,res(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.3659662,res(1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.8380736,res(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.12308775,res(1).d_.val_.val());

  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res(0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.045198753,h[0]);
  EXPECT_FLOAT_EQ(-0.068486936,h[1]);
  EXPECT_FLOAT_EQ(-0.011624861,h[2]);
  EXPECT_FLOAT_EQ(0.027791996,h[3]);
}
