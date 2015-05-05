#include <stan/math/prim/mat/fun/singular_values.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

TEST(AgradMixMatrixSingularValues, mat_fv_1st_deriv) {
  stan::math::matrix_fv m1(2,2);
  stan::math::vector_fv res;

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
TEST(AgradMixMatrixSingularValues, mat_fv_2nd_deriv) {
  stan::math::matrix_fv m1(2,2);
  stan::math::vector_fv res;

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

TEST(AgradMixMatrixSingularValues, mat_ffv_1st_deriv) {
  stan::math::matrix_ffv m1(2,2);
  stan::math::vector_ffv res;

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
TEST(AgradMixMatrixSingularValues, mat_ffv_2nd_deriv) {
  stan::math::matrix_ffv m1(2,2);
  stan::math::vector_ffv res;

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

TEST(AgradMixMatrixSingularValues, mat_ffv_3rd_deriv) {
  stan::math::matrix_ffv m1(2,2);
  stan::math::vector_ffv res;

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
