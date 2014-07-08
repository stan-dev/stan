#include <stdexcept>
#include <stan/math/matrix/head.hpp>

#include <gtest/gtest.h>

TEST(MathMatrixHead,HeadVector1) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(MathMatrixHead,HeadVector2) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(MathMatrixHead,HeadVector3) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixHead,HeadVector4) {
  using stan::math::head;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;

  Eigen::VectorXd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}

TEST(MathMatrixHead,HeadRowVector1) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(MathMatrixHead,HeadRowVector2) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(MathMatrixHead,HeadRowVector3) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixHead,HeadRowVector4) {
  using stan::math::head;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;

  Eigen::RowVectorXd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}


TEST(MathMatrixHead,HeadStdVector1) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(MathMatrixHead,HeadStdVector2) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(MathMatrixHead,HeadStdVector3) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(MathMatrixHead,HeadStdVector4) {
  using stan::math::head;
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<int> v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n],v01[n]);
}
