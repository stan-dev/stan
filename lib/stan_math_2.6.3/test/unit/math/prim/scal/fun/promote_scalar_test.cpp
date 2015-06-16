#include <stan/math/prim/mat/fun/promote_scalar.hpp>
#include <test/unit/math/prim/scal/fun/promote_type_test_util.hpp>
#include <gtest/gtest.h>

TEST(MathFunctionsPromoteScalar, Match) {
  using stan::math::promote_scalar;
  EXPECT_FLOAT_EQ(1.3, promote_scalar<double>(1.3));
  EXPECT_EQ(3, promote_scalar<int>(3));

  expect_type<double>(promote_scalar<double>(2.3));
  expect_type<int>(promote_scalar<int>(2));
}
TEST(MathFunctionsPromoteScalar, Mismatch) {
  using stan::math::promote_scalar;
  EXPECT_FLOAT_EQ(2.0, promote_scalar<double>(2));
  expect_type<double>(promote_scalar<double>(2));
}
TEST(MathFunctionsPromoteScalar, VectorMismatch) {
  using stan::math::promote_scalar;
  std::vector<int> x;
  x.push_back(1);
  x.push_back(2);
  std::vector<double> y = promote_scalar<double>(x);
  EXPECT_EQ(2U, y.size());
  EXPECT_FLOAT_EQ(1.0, y[0]);
  EXPECT_FLOAT_EQ(2.0, y[1]);
}
TEST(MathFunctionsPromoteScalar, VectorMatch) {
  using stan::math::promote_scalar;
  std::vector<int> x;
  x.push_back(13);
  x.push_back(27);
  std::vector<int> y = promote_scalar<int>(x);
  EXPECT_EQ(2U, y.size());
  EXPECT_EQ(13, y[0]);
  EXPECT_EQ(27, y[1]);
}
TEST(MathFunctionsPromoteScalar, Vector2Mismatch) {
  using stan::math::promote_scalar;
  using std::vector;
  vector<vector<int> > x(2);
  x[0].push_back(1);
  x[0].push_back(2);
  x[0].push_back(3);
  x[1].push_back(4);
  x[1].push_back(5);
  x[1].push_back(6);

  vector<vector<double> > y = promote_scalar<double>(x);
  EXPECT_EQ(2U, y.size());
  EXPECT_EQ(3, y[0].size());
  EXPECT_FLOAT_EQ(1.0, y[0][0]);
  EXPECT_FLOAT_EQ(2.0, y[0][1]);
  EXPECT_FLOAT_EQ(3.0, y[0][2]);
  EXPECT_FLOAT_EQ(4.0, y[1][0]);
  EXPECT_FLOAT_EQ(5.0, y[1][1]);
  EXPECT_FLOAT_EQ(6.0, y[1][2]);
}
TEST(MathFunctionsPromoteScalar, Vector2Match) {
  using stan::math::promote_scalar;
  using std::vector;
  vector<vector<double> > x(2);
  x[0].push_back(1.1);
  x[0].push_back(2.2);
  x[0].push_back(3.3);
  x[1].push_back(4.4);
  x[1].push_back(5.5);
  x[1].push_back(6.6);

  vector<vector<double> > y = promote_scalar<double>(x);
  EXPECT_EQ(2U, y.size());
  EXPECT_EQ(3, y[0].size());
  EXPECT_FLOAT_EQ(1.1, y[0][0]);
  EXPECT_FLOAT_EQ(2.2, y[0][1]);
  EXPECT_FLOAT_EQ(3.3, y[0][2]);
  EXPECT_FLOAT_EQ(4.4, y[1][0]);
  EXPECT_FLOAT_EQ(5.5, y[1][1]);
  EXPECT_FLOAT_EQ(6.6, y[1][2]);
}
