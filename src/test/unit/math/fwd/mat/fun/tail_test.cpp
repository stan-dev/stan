#include <stdexcept>
#include <stan/math/prim/mat/fun/tail.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;

TEST(AgradFwdMatrixTail,TailVector1_fd) {
  using stan::math::tail;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_fd) {
  using stan::math::tail;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_fd) {
  using stan::math::tail;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,TailVector4_fd) {
  using stan::math::tail;
  stan::math::vector_fd v(3);
  v << 1, 2, 3;

  stan::math::vector_fd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_,v12[n].val_);
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_fd) {
  using stan::math::tail;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_fd) {
  using stan::math::tail;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_fd) {
  using stan::math::tail;
  stan::math::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,tailRowVector4_fd) {
  using stan::math::tail;
   stan::math::row_vector_fd v(3);
  v << 1, 2, 3;

  stan::math::row_vector_fd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_,v12[n].val_);
  }
}


TEST(AgradFwdMatrixTail,tailStdVector1_fd) {
  using stan::math::tail;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailStdVector2_fd) {
  using stan::math::tail;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailStdVector3_fd) {
  using stan::math::tail;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,tailStdVector4_fd) {
  using stan::math::tail;
  std::vector<fvar<double> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<double> > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_,v12[n].val_);
  }
}
TEST(AgradFwdMatrixTail,TailVector1_ffd) {
  using stan::math::tail;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_ffd) {
  using stan::math::tail;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_ffd) {
  using stan::math::tail;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,TailVector4_ffd) {
  using stan::math::tail;
  stan::math::vector_ffd v(3);
  v << 1, 2, 3;

  stan::math::vector_ffd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v12[n].val_.val_);
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_ffd) {
  using stan::math::tail;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_ffd) {
  using stan::math::tail;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_ffd) {
  using stan::math::tail;
  stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,tailRowVector4_ffd) {
  using stan::math::tail;
   stan::math::row_vector_ffd v(3);
  v << 1, 2, 3;

  stan::math::row_vector_ffd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v12[n].val_.val_);
  }
}


TEST(AgradFwdMatrixTail,tailStdVector1_ffd) {
  using stan::math::tail;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailStdVector2_ffd) {
  using stan::math::tail;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailStdVector3_ffd) {
  using stan::math::tail;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradFwdMatrixTail,tailStdVector4_ffd) {
  using stan::math::tail;
  std::vector<fvar<fvar<double> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<double> > > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v12[n].val_.val_);
  }
}
