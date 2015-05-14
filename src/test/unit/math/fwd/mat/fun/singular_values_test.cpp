#include <stan/math/prim/mat/fun/singular_values.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

TEST(AgradFwdMatrixSingularValues, mat_fd) {
  stan::math::matrix_fd m0(2,2);
  stan::math::vector_fd res;

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
  stan::math::matrix_ffd m0(2,2);
  stan::math::vector_ffd res;

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

