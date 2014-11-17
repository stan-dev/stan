#include <stdexcept>
#include <stan/math/matrix/tail.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixTail,TailVector1_fd) {
  using stan::math::tail;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_fd) {
  using stan::math::tail;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_fd) {
  using stan::math::tail;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,TailVector4_fd) {
  using stan::math::tail;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;

  stan::agrad::vector_fd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_,v12[n].val_);
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_fd) {
  using stan::math::tail;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_fd) {
  using stan::math::tail;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_fd) {
  using stan::math::tail;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailRowVector4_fd) {
  using stan::math::tail;
   stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_fd v12 = tail(v,2);
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
  EXPECT_THROW(tail(v,4), std::domain_error);
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
TEST(AgradFwdMatrixTail,TailVector1_fv) {
  using stan::math::tail;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_fv) {
  using stan::math::tail;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_fv) {
  using stan::math::tail;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,TailVector4_fv) {
  using stan::math::tail;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;

  stan::agrad::vector_fv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_fv) {
  using stan::math::tail;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_fv) {
  using stan::math::tail;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_fv) {
  using stan::math::tail;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailRowVector4_fv) {
  using stan::math::tail;
   stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_fv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}


TEST(AgradFwdMatrixTail,tailStdVector1_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailStdVector2_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailStdVector3_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailStdVector4_fv) {
  using stan::math::tail;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<var> > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val(),v12[n].val_.val());
  }
}
TEST(AgradFwdMatrixTail,TailVector1_ffd) {
  using stan::math::tail;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_ffd) {
  using stan::math::tail;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_ffd) {
  using stan::math::tail;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,TailVector4_ffd) {
  using stan::math::tail;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;

  stan::agrad::vector_ffd v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_,v12[n].val_.val_);
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_ffd) {
  using stan::math::tail;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_ffd) {
  using stan::math::tail;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_ffd) {
  using stan::math::tail;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailRowVector4_ffd) {
  using stan::math::tail;
   stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_ffd v12 = tail(v,2);
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
  EXPECT_THROW(tail(v,4), std::domain_error);
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
TEST(AgradFwdMatrixTail,TailVector1_ffv) {
  using stan::math::tail;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,TailVector2_ffv) {
  using stan::math::tail;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,TailVector3_ffv) {
  using stan::math::tail;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,TailVector4_ffv) {
  using stan::math::tail;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;

  stan::agrad::vector_ffv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}

TEST(AgradFwdMatrixTail,tailRowVector1_ffv) {
  using stan::math::tail;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailRowVector2_ffv) {
  using stan::math::tail;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailRowVector3_ffv) {
  using stan::math::tail;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailRowVector4_ffv) {
  using stan::math::tail;
   stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_ffv v12 = tail(v,2);
  EXPECT_EQ(2,v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}


TEST(AgradFwdMatrixTail,tailStdVector1_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, tail(v,0).size());
}
TEST(AgradFwdMatrixTail,tailStdVector2_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, tail(v,3).size());
}
TEST(AgradFwdMatrixTail,tailStdVector3_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(tail(v,4), std::domain_error);
}
TEST(AgradFwdMatrixTail,tailStdVector4_ffv) {
  using stan::math::tail;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<var> > > v12 = tail(v,2);
  EXPECT_EQ(2U, v12.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n+1].val_.val_.val(),v12[n].val_.val_.val());
  }
}
