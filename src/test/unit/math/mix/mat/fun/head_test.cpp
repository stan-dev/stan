#include <stdexcept>
#include <stan/math/prim/mat/fun/head.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixHead,HeadVector1_fv) {
  using stan::math::head;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadVector2_fv) {
  using stan::math::head;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadVector3_fv) {
  using stan::math::head;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadVector4_fv) {
  using stan::math::head;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;

  stan::math::vector_fv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}

TEST(AgradMixMatrixHead,HeadRowVector1_fv) {
  using stan::math::head;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadRowVector2_fv) {
  using stan::math::head;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadRowVector3_fv) {
  using stan::math::head;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadRowVector4_fv) {
  using stan::math::head;
   stan::math::row_vector_fv v(3);
  v << 1, 2, 3;

  stan::math::row_vector_fv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}


TEST(AgradMixMatrixHead,HeadStdVector1_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadStdVector2_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadStdVector3_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadStdVector4_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<var> > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}
TEST(AgradMixMatrixHead,HeadVector1_ffv) {
  using stan::math::head;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadVector2_ffv) {
  using stan::math::head;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadVector3_ffv) {
  using stan::math::head;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadVector4_ffv) {
  using stan::math::head;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;

  stan::math::vector_ffv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}

TEST(AgradMixMatrixHead,HeadRowVector1_ffv) {
  using stan::math::head;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadRowVector2_ffv) {
  using stan::math::head;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadRowVector3_ffv) {
  using stan::math::head;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadRowVector4_ffv) {
  using stan::math::head;
   stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;

  stan::math::row_vector_ffv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}


TEST(AgradMixMatrixHead,HeadStdVector1_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradMixMatrixHead,HeadStdVector2_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradMixMatrixHead,HeadStdVector3_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradMixMatrixHead,HeadStdVector4_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<var> > > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}
