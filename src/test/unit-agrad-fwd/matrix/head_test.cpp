#include <stdexcept>
#include <stan/math/matrix/head.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixHead,HeadVector1_fd) {
  using stan::math::head;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_fd) {
  using stan::math::head;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_fd) {
  using stan::math::head;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadVector4_fd) {
  using stan::math::head;
  stan::agrad::vector_fd v(3);
  v << 1, 2, 3;

  stan::agrad::vector_fd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_,v01[n].val_);
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_fd) {
  using stan::math::head;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_fd) {
  using stan::math::head;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_fd) {
  using stan::math::head;
  stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_fd) {
  using stan::math::head;
   stan::agrad::row_vector_fd v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_fd v01 = head(v,2);
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
  EXPECT_THROW(head(v,4), std::domain_error);
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
TEST(AgradFwdMatrixHead,HeadVector1_fv) {
  using stan::math::head;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_fv) {
  using stan::math::head;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_fv) {
  using stan::math::head;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadVector4_fv) {
  using stan::math::head;
  stan::agrad::vector_fv v(3);
  v << 1, 2, 3;

  stan::agrad::vector_fv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_fv) {
  using stan::math::head;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_fv) {
  using stan::math::head;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_fv) {
  using stan::math::head;
  stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_fv) {
  using stan::math::head;
   stan::agrad::row_vector_fv v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_fv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}


TEST(AgradFwdMatrixHead,HeadStdVector1_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector2_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector3_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadStdVector4_fv) {
  using stan::math::head;
  std::vector<fvar<var> > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<var> > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val(),v01[n].val_.val());
  }
}
TEST(AgradFwdMatrixHead,HeadVector1_ffd) {
  using stan::math::head;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_ffd) {
  using stan::math::head;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_ffd) {
  using stan::math::head;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadVector4_ffd) {
  using stan::math::head;
  stan::agrad::vector_ffd v(3);
  v << 1, 2, 3;

  stan::agrad::vector_ffd v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_,v01[n].val_.val_);
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_ffd) {
  using stan::math::head;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_ffd) {
  using stan::math::head;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_ffd) {
  using stan::math::head;
  stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_ffd) {
  using stan::math::head;
   stan::agrad::row_vector_ffd v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_ffd v01 = head(v,2);
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
  EXPECT_THROW(head(v,4), std::domain_error);
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
TEST(AgradFwdMatrixHead,HeadVector1_ffv) {
  using stan::math::head;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadVector2_ffv) {
  using stan::math::head;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadVector3_ffv) {
  using stan::math::head;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadVector4_ffv) {
  using stan::math::head;
  stan::agrad::vector_ffv v(3);
  v << 1, 2, 3;

  stan::agrad::vector_ffv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}

TEST(AgradFwdMatrixHead,HeadRowVector1_ffv) {
  using stan::math::head;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(0, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector2_ffv) {
  using stan::math::head;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_EQ(3, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadRowVector3_ffv) {
  using stan::math::head;
  stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadRowVector4_ffv) {
  using stan::math::head;
   stan::agrad::row_vector_ffv v(3);
  v << 1, 2, 3;

  stan::agrad::row_vector_ffv v01 = head(v,2);
  EXPECT_EQ(2,v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}


TEST(AgradFwdMatrixHead,HeadStdVector1_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(0U, head(v,0).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector2_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_EQ(3U, head(v,3).size());
}
TEST(AgradFwdMatrixHead,HeadStdVector3_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  EXPECT_THROW(head(v,4), std::domain_error);
}
TEST(AgradFwdMatrixHead,HeadStdVector4_ffv) {
  using stan::math::head;
  std::vector<fvar<fvar<var> > > v;
  v.push_back(1); v.push_back(2); v.push_back(3);
  std::vector<fvar<fvar<var> > > v01 = head(v,2);
  EXPECT_EQ(2U, v01.size());
  for (int n = 0; n < 2; ++n) {
    EXPECT_FLOAT_EQ(v[n].val_.val_.val(),v01[n].val_.val_.val());
  }
}
