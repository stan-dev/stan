#include <stdexcept>
#include <stan/math/matrix/head.hpp>

#include <gtest/gtest.h>

TEST(MathMatrixBlock,HeadVector1) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(MathMatrixBlock,HeadVector2) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(MathMatrixBlock,HeadVector3) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixBlock,HeadVector4) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;

  Eigen::VectorXd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}

TEST(MathMatrixBlock,HeadRowVector1) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(MathMatrixBlock,HeadRowVector2) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(MathMatrixBlock,HeadRowVector3) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixBlock,HeadRowVector4) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;

  Eigen::RowVectorXd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}


TEST(MathMatrixBlock,HeadStdVector1) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0, head(v,0).size());
}
TEST(MathMatrixBlock,HeadStdVector2) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3, head(v,3).size());
}
TEST(MathMatrixBlock,HeadStdVector3) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixBlock,HeadStdVector4) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<int> v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}
