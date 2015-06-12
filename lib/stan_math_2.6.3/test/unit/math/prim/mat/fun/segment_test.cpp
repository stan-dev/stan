#include <stdexcept>
#include <stan/math/prim/mat/fun/segment.hpp>
#include <gtest/gtest.h>


TEST(MathMatrixSegment,SegmentVector1) {
  using stan::math::segment;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(MathMatrixSegment,SegmentVector2) {
  using stan::math::segment;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(MathMatrixSegment,SegmentVector3) {
  using stan::math::segment;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(MathMatrixSegment,SegmentVector4) {
  using stan::math::segment;
  Eigen::VectorXd v(4);
  v << 1, 2, 3, 4;

  Eigen::VectorXd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v23[n]);
}

TEST(MathMatrixSegment,SegmentRowVector1) {
  using stan::math::segment;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(MathMatrixSegment,SegmentRowVector2) {
  using stan::math::segment;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(MathMatrixSegment,SegmentRowVector3) {
  using stan::math::segment;
  Eigen::RowVectorXd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(MathMatrixSegment,SegmentRowVector4) {
  using stan::math::segment;
  Eigen::RowVectorXd v(4);
  v << 1, 2, 3, 4;

  Eigen::RowVectorXd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v23[n]);
}

TEST(MathMatrixSegment,SegmentStdVector1) {
  using stan::math::segment;
  std::vector<int> v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(MathMatrixSegment,SegmentStdVector2) {
  using stan::math::segment;
  std::vector<int> v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(MathMatrixSegment,SegmentStdVector3) {
  using stan::math::segment;
  std::vector<int> v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(MathMatrixSegment,SegmentStdVector4) {
  using stan::math::segment;
  std::vector<int> v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<int> v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1],v23[n]);
}
