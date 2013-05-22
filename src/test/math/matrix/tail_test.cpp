#include <stdexcept>
#include <stan/math/matrix/tail.hpp>

#include <gtest/gtest.h>

TEST(MathMatrixBlock,TailVector1) {
  using stan::math::tail;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(MathMatrixBlock,TailVector2) {
  using stan::math::tail;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(MathMatrixBlock,TailVector3) {
  using stan::math::tail;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(MathMatrixBlock,TailVector4) {
  using stan::math::tail;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;

  Eigen::VectorXd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v12[n]);
}

TEST(MathMatrixBlock,TailRowVector1) {
  using stan::math::tail;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(MathMatrixBlock,TailRowVector2) {
  using stan::math::tail;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(MathMatrixBlock,TailRowVector3) {
  using stan::math::tail;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(MathMatrixBlock,TailRowVector4) {
  using stan::math::tail;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  Eigen::RowVectorXd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v12[n]);
}


TEST(MathMatrixBlock,TailStdVector1) {
  using stan::math::tail;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(MathMatrixBlock,TailStdVector2) {
  using stan::math::tail;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(MathMatrixBlock,TailStdVector3) {
  using stan::math::tail;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(MathMatrixBlock,TailStdVector4) {
  using stan::math::tail;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<int> v12 = tail(v,2);
  EXPECT_EQ(2U,v12.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v12[n]);
}
