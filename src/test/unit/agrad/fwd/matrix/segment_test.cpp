#include <stdexcept>
#include <stan/math/matrix/segment.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>
using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixSegment,SegmentVector1_fd) {
  using stan::math::segment;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_fd) {
  using stan::math::segment;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_fd) {
  using stan::math::segment;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_fd) {
  using stan::math::segment;
  stan::agrad::vector_fd v(4);
  v << 1, 2, 3, 4;

  stan::agrad::vector_fd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_,v23[n].val_);
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_fd) {
  using stan::math::segment;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_fd) {
  using stan::math::segment;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_fd) {
  using stan::math::segment;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_fd) {
  using stan::math::segment;
  stan::agrad::row_vector_fd v(4);
  v << 1, 2, 3, 4;

  stan::agrad::row_vector_fd v23 = segment(v,2,2);
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

TEST(AgradFwdMatrixSegment,SegmentVector1_fv) {
  using stan::math::segment;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_fv) {
  using stan::math::segment;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_fv) {
  using stan::math::segment;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_fv) {
  using stan::math::segment;
  stan::agrad::vector_fv v(4);
  v << 1, 2, 3, 4;

  stan::agrad::vector_fv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_fv) {
  using stan::math::segment;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_fv) {
  using stan::math::segment;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_fv) {
  using stan::math::segment;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_fv) {
  using stan::math::segment;
  stan::agrad::row_vector_fv v(4);
  v << 1, 2, 3, 4;

  stan::agrad::row_vector_fv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}

TEST(AgradFwdMatrixSegment,SegmentStdVector1_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector2_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector3_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentStdVector4_fv) {
  using stan::math::segment;
  std::vector<fvar<var> > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<var> > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v23[n].val_.val());
}
TEST(AgradFwdMatrixSegment,SegmentVector1_ffd) {
  using stan::math::segment;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_ffd) {
  using stan::math::segment;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_ffd) {
  using stan::math::segment;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_ffd) {
  using stan::math::segment;
  stan::agrad::vector_ffd v(4);
  v << 1, 2, 3, 4;

  stan::agrad::vector_ffd v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v23[n].val_.val_);
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_ffd) {
  using stan::math::segment;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_ffd) {
  using stan::math::segment;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_ffd) {
  using stan::math::segment;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_ffd) {
  using stan::math::segment;
  stan::agrad::row_vector_ffd v(4);
  v << 1, 2, 3, 4;

  stan::agrad::row_vector_ffd v23 = segment(v,2,2);
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

TEST(AgradFwdMatrixSegment,SegmentVector1_ffv) {
  using stan::math::segment;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector2_ffv) {
  using stan::math::segment;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentVector3_ffv) {
  using stan::math::segment;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentVector4_ffv) {
  using stan::math::segment;
  stan::agrad::vector_ffv v(4);
  v << 1, 2, 3, 4;

  stan::agrad::vector_ffv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}

TEST(AgradFwdMatrixSegment,SegmentRowVector1_ffv) {
  using stan::math::segment;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector2_ffv) {
  using stan::math::segment;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentRowVector3_ffv) {
  using stan::math::segment;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(segment(v,1,4), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentRowVector4_ffv) {
  using stan::math::segment;
  stan::agrad::row_vector_ffv v(4);
  v << 1, 2, 3, 4;

  stan::agrad::row_vector_ffv v23 = segment(v,2,2);
  EXPECT_EQ(2,v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}

TEST(AgradFwdMatrixSegment,SegmentStdVector1_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(0U, segment(v,1,0).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector2_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_EQ(3U, segment(v,1,3).size());
}
TEST(AgradFwdMatrixSegment,SegmentStdVector3_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  EXPECT_THROW(segment(v,1,7), std::domain_error);
}
TEST(AgradFwdMatrixSegment,SegmentStdVector4_ffv) {
  using stan::math::segment;
  std::vector<fvar<fvar<var> > > v(3);
  v.push_back(1);  v.push_back(2);  v.push_back(3);
  std::vector<fvar<fvar<var> > > v23 = segment(v,2,2);
  EXPECT_EQ(2U, v23.size());
  for (int n = 0; n < 2; ++n)
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v23[n].val_.val_.val());
}


