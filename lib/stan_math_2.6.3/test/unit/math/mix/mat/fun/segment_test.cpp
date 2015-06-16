#include <stdexcept>
#include <stan/math/prim/mat/fun/segment.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixSegment,SegmentVector1_fv) {
  using stan::math::segment;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentVector2_fv) {
  using stan::math::segment;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentVector3_fv) {
  using stan::math::segment;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentVector4_fv) {
  using stan::math::segment;
  stan::math::vector_fv v(4);
  v << 1, 2, 3, 4;

  stan::math::vector_fv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}

TEST(AgradMixMatrixSegment,SegmentRowVector1_fv) {
  using stan::math::segment;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentRowVector2_fv) {
  using stan::math::segment;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentRowVector3_fv) {
  using stan::math::segment;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentRowVector4_fv) {
  using stan::math::segment;
  stan::math::row_vector_fv v(4);
  v << 1, 2, 3, 4;

  stan::math::row_vector_fv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}

TEST(AgradMixMatrixSegment,SegmentStdVector1_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentStdVector2_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentStdVector3_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentStdVector4_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<var> > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}
TEST(AgradMixMatrixSegment,SegmentVector1_ffv) {
  using stan::math::segment;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentVector2_ffv) {
  using stan::math::segment;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentVector3_ffv) {
  using stan::math::segment;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentVector4_ffv) {
  using stan::math::segment;
  stan::math::vector_ffv v(4);
  v << 1, 2, 3, 4;

  stan::math::vector_ffv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}

TEST(AgradMixMatrixSegment,SegmentRowVector1_ffv) {
  using stan::math::segment;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentRowVector2_ffv) {
  using stan::math::segment;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentRowVector3_ffv) {
  using stan::math::segment;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentRowVector4_ffv) {
  using stan::math::segment;
  stan::math::row_vector_ffv v(4);
  v << 1, 2, 3, 4;

  stan::math::row_vector_ffv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}

TEST(AgradMixMatrixSegment,SegmentStdVector1_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradMixMatrixSegment,SegmentStdVector2_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradMixMatrixSegment,SegmentStdVector3_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradMixMatrixSegment,SegmentStdVector4_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<fvar<var> > > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}


