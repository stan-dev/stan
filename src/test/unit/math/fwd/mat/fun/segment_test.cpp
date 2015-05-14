#include <stdexcept>
#include <stan/math/prim/mat/fun/segment.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <gtest/gtest.h>
using stan::math::fvar;

TEST(AgradFwdMatrixSegment,SegmentVector1_fd) {
  using stan::math::segment;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_fd) {
  using stan::math::segment;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_fd) {
  using stan::math::segment;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_fd) {
  using stan::math::segment;
  stan::math::vector_fd v(4);
  v << 1, 2, 3, 4;

  stan::math::vector_fd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_,v23[n].val_);
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_fd) {
  using stan::math::segment;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_fd) {
  using stan::math::segment;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_fd) {
  using stan::math::segment;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_fd) {
  using stan::math::segment;
  stan::math::row_vector_fd v(4);
  v << 1, 2, 3, 4;

  stan::math::row_vector_fd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_,v23[n].val_);
}

TEST(AgradFwdMatrixSegment,SegmentStdVector1_fd) {
  using stan::math::segment;
  std::vector<fvar<double> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector2_fd) {
  using stan::math::segment;
  std::vector<fvar<double> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector3_fd) {
  using stan::math::segment;
  std::vector<fvar<double> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentStdVector4_fd) {
  using stan::math::segment;
  std::vector<fvar<double> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<double> > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_,v23[n].val_);
}

TEST(AgradFwdMatrixSegment,SegmentVector1_ffd) {
  using stan::math::segment;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_ffd) {
  using stan::math::segment;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_ffd) {
  using stan::math::segment;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_ffd) {
  using stan::math::segment;
  stan::math::vector_ffd v(4);
  v << 1, 2, 3, 4;

  stan::math::vector_ffd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v23[n].val_.val_);
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_ffd) {
  using stan::math::segment;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_ffd) {
  using stan::math::segment;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_ffd) {
  using stan::math::segment;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_ffd) {
  using stan::math::segment;
  stan::math::row_vector_ffd v(4);
  v << 1, 2, 3, 4;

  stan::math::row_vector_ffd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v23[n].val_.val_);
}

TEST(AgradFwdMatrixSegment,SegmentStdVector1_ffd) {
  using stan::math::segment;
  std::vector<fvar<fvar<double> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector2_ffd) {
  using stan::math::segment;
  std::vector<fvar<fvar<double> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector3_ffd) {
  using stan::math::segment;
  std::vector<fvar<fvar<double> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentStdVector4_ffd) {
  using stan::math::segment;
  std::vector<fvar<fvar<double> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<fvar<double> > > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v23[n].val_.val_);
}
