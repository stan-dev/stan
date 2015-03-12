#include <stan/math/prim/mat/meta/index_type.hpp>
#include <test/unit/math/prim/scal/fun/promote_type_test_util.hpp>
#include <gtest/gtest.h>

TEST(MathMeta, index_type) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::index_type;

  expect_same_type<Matrix<double,Dynamic,Dynamic>::Index,
                   index_type<Matrix<double,Dynamic,Dynamic> >::type>();

  expect_same_type<Matrix<double,Dynamic,1>::Index,
                   index_type<Matrix<double,Dynamic,1> >::type>();

  expect_same_type<Matrix<double,1,Dynamic>::Index,
                   index_type<Matrix<double,1,Dynamic> >::type>();
}
