#include <stdexcept>
#include <stan/math/prim/mat/fun/tail.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixTail,TailVector1_fv) {
  using stan::math::tail;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradMixMatrixTail,TailVector2_fv) {
  using stan::math::tail;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradMixMatrixTail,TailVector3_fv) {
  using stan::math::tail;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,TailVector4_fv) {
  using stan::math::tail;
  stan::math::vector_fv v(3);
  v << 1, 2, 3;

  stan::math::vector_fv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}

TEST(AgradMixMatrixTail,tailRowVector1_fv) {
  using stan::math::tail;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradMixMatrixTail,tailRowVector2_fv) {
  using stan::math::tail;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradMixMatrixTail,tailRowVector3_fv) {
  using stan::math::tail;
  stan::math::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,tailRowVector4_fv) {
  using stan::math::tail;
   stan::math::row_vector_fv v(3);
  v << 1, 2, 3;

  stan::math::row_vector_fv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}


TEST(AgradMixMatrixTail,tailStdVector1_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradMixMatrixTail,tailStdVector2_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradMixMatrixTail,tailStdVector3_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,tailStdVector4_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<var> > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}
TEST(AgradMixMatrixTail,TailVector1_ffv) {
  using stan::math::tail;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradMixMatrixTail,TailVector2_ffv) {
  using stan::math::tail;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradMixMatrixTail,TailVector3_ffv) {
  using stan::math::tail;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,TailVector4_ffv) {
  using stan::math::tail;
  stan::math::vector_ffv v(3);
  v << 1, 2, 3;

  stan::math::vector_ffv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}

TEST(AgradMixMatrixTail,tailRowVector1_ffv) {
  using stan::math::tail;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradMixMatrixTail,tailRowVector2_ffv) {
  using stan::math::tail;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradMixMatrixTail,tailRowVector3_ffv) {
  using stan::math::tail;
  stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,tailRowVector4_ffv) {
  using stan::math::tail;
   stan::math::row_vector_ffv v(3);
  v << 1, 2, 3;

  stan::math::row_vector_ffv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}


TEST(AgradMixMatrixTail,tailStdVector1_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradMixMatrixTail,tailStdVector2_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradMixMatrixTail,tailStdVector3_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::out_of_range);
}
TEST(AgradMixMatrixTail,tailStdVector4_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<var> > > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}
