#include <stdexcept>
#include <stan/math/prim/mat/fun/head.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;

TEST(AgradFwdMatrixHead,HeadVector1_fd) {
  using stan::math::head;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_fd) {
  using stan::math::head;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_fd) {
  using stan::math::head;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadVector4_fd) {
  using stan::math::head;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;

  stan::math::vector_fd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_,v01[n].val_);
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_fd) {
  using stan::math::head;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_fd) {
  using stan::math::head;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_fd) {
  using stan::math::head;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_fd) {
  using stan::math::head;
   stan::math::row_vector_fd v(3);
  v << 1, 2, 3;

  stan::math::row_vector_fd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_,v01[n].val_);
  }
}


TEST(AgradFwdMatrixHead,HeadStdVector1_fd) {
  using stan::math::head;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector2_fd) {
  using stan::math::head;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector3_fd) {
  using stan::math::head;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadStdVector4_fd) {
  using stan::math::head;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<double> > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_,v01[n].val_);
  }
}
TEST(AgradFwdMatrixHead,HeadVector1_ffd) {
  using stan::math::head;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_ffd) {
  using stan::math::head;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_ffd) {
  using stan::math::head;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadVector4_ffd) {
  using stan::math::head;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;

  stan::math::vector_ffd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_,v01[n].val_.val_);
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_ffd) {
  using stan::math::head;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_ffd) {
  using stan::math::head;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_ffd) {
  using stan::math::head;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_ffd) {
  using stan::math::head;
   stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;

  stan::math::row_vector_ffd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_,v01[n].val_.val_);
  }
}


TEST(AgradFwdMatrixHead,HeadStdVector1_ffd) {
  using stan::math::head;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector2_ffd) {
  using stan::math::head;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector3_ffd) {
  using stan::math::head;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixHead,HeadStdVector4_ffd) {
  using stan::math::head;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<double> > > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_,v01[n].val_.val_);
  }
}
