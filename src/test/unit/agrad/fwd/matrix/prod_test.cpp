#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/prod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixProd,fd_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;
  using stan::agrad::fvar;

  vector_d vd;
  vector_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = vector_d(1);
  vv = vector_fd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_fd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}

TEST(AgradFwdMatrixProd,fd_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;
  using stan::agrad::fvar;

  row_vector_d vd;
  row_vector_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = row_vector_d(1);
  vv = row_vector_fd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_fd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}
TEST(AgradFwdMatrixProd,fd_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;

  matrix_d vd;
  matrix_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = matrix_d(1,1);
  vv = matrix_fd(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_fd(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_);
  EXPECT_FLOAT_EQ(60.0,f.d_);
}
TEST(AgradFwdMatrixProd,fv_vector_1stDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d vd;
  vector_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = vector_d(1);
  vv = vector_fv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_fv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  std::vector<fvar<var> > x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());

  AVEC q = createAVEC(vv(0).val(),vv(1).val());
  VEC h;
  f.val_.grad(q,h);
  EXPECT_FLOAT_EQ(3,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
}
TEST(AgradFwdMatrixProd,fv_vector_2ndDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_fv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<var> f = prod(vv);

  AVEC q = createAVEC(vv(0).val(),vv(1).val());
  VEC h;
  f.d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
}

TEST(AgradFwdMatrixProd,fv_rowvector_1stDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d vd;
  row_vector_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = row_vector_d(1);
  vv = row_vector_fv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_fv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());

  AVEC q = createAVEC(vv(0).val(),vv(1).val());
  VEC h;
  f.val_.grad(q,h);
  EXPECT_FLOAT_EQ(3,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
}
TEST(AgradFwdMatrixProd,fv_rowvector_2ndDeriv) {
  using stan::math::prod;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_fv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<var> f = prod(vv);

  AVEC q = createAVEC(vv(0).val(),vv(1).val());
  VEC h;
  f.d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
}
TEST(AgradFwdMatrixProd,fv_matrix_1stDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d vd;
  matrix_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = matrix_d(1,1);
  vv = matrix_fv(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_fv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_.val());
  EXPECT_FLOAT_EQ(60.0,f.d_.val());

  AVEC q = createAVEC(vv(0,0).val(),vv(0,1).val(),vv(1,0).val(),vv(1,1).val());
  VEC h;
  f.val_.grad(q,h);
  EXPECT_FLOAT_EQ(18,h[0]);
  EXPECT_FLOAT_EQ(12,h[1]);
  EXPECT_FLOAT_EQ(18,h[2]);
  EXPECT_FLOAT_EQ(12,h[3]);
}
TEST(AgradFwdMatrixProd,fv_matrix_2ndDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_fv vv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<var> f = prod(vv);

  AVEC q = createAVEC(vv(0,0).val(),vv(0,1).val(),vv(1,0).val(),vv(1,1).val());
  VEC h;
  f.d_.grad(q,h);
  EXPECT_FLOAT_EQ(21,h[0]);
  EXPECT_FLOAT_EQ(16,h[1]);
  EXPECT_FLOAT_EQ(21,h[2]);
  EXPECT_FLOAT_EQ(16,h[3]);
}
TEST(AgradFwdMatrixProd,ffd_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d vd;
  vector_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = vector_d(1);
  vv = vector_ffd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_ffd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}

TEST(AgradFwdMatrixProd,ffd_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d vd;
  row_vector_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = row_vector_d(1);
  vv = row_vector_ffd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_ffd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  row_vector_ffd x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}
TEST(AgradFwdMatrixProd,ffd_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d vd;
  matrix_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = matrix_d(1,1);
  vv = matrix_ffd(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_ffd(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_.val());
  EXPECT_FLOAT_EQ(60.0,f.d_.val());
}
TEST(AgradFwdMatrixProd,ffv_vector_1stDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d vd;
  vector_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val().val());

  vd = vector_d(1);
  vv = vector_ffv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val().val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_ffv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  std::vector<fvar<fvar<var> > > x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<fvar<var> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val().val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val().val());

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(3,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_vector_2ndDeriv_1) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_vector_2ndDeriv_2) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_vector_3rdDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
   vv(0).val_.d_ = 1.0;
   vv(1).val_.d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}

TEST(AgradFwdMatrixProd,ffv_rowvector_1stDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d vd;
  row_vector_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val().val());

  vd = row_vector_d(1);
  vv = row_vector_ffv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val().val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_ffv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val().val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val().val());

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(3,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_rowvector_2ndDeriv_1) {
  using stan::math::prod;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_rowvector_2ndDeriv_2) {
  using stan::math::prod;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(1,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_rowvector_3rdDeriv) {
  using stan::math::prod;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_ffv vv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
   vv(0).val_.d_ = 1.0;
   vv(1).val_.d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0).val().val(),vv(1).val().val());
  VEC h;
  f.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
}
TEST(AgradFwdMatrixProd,ffv_matrix_1stDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d vd;
  matrix_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val().val());

  vd = matrix_d(1,1);
  vv = matrix_ffv(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val().val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val().val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_ffv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_.val().val());
  EXPECT_FLOAT_EQ(60.0,f.d_.val().val());

  AVEC q = createAVEC(vv(0,0).val().val(),vv(0,1).val().val(),vv(1,0).val().val(),vv(1,1).val().val());
  VEC h;
  f.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(18,h[0]);
  EXPECT_FLOAT_EQ(12,h[1]);
  EXPECT_FLOAT_EQ(18,h[2]);
  EXPECT_FLOAT_EQ(12,h[3]);
}
TEST(AgradFwdMatrixProd,ffv_matrix_2ndDeriv_1) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_ffv vv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0,0).val().val(),vv(0,1).val().val(),vv(1,0).val().val(),vv(1,1).val().val());
  VEC h;
  f.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradFwdMatrixProd,ffv_matrix_2ndDeriv_2) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_ffv vv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0,0).val().val(),vv(0,1).val().val(),vv(1,0).val().val(),vv(1,1).val().val());
  VEC h;
  f.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(21,h[0]);
  EXPECT_FLOAT_EQ(16,h[1]);
  EXPECT_FLOAT_EQ(21,h[2]);
  EXPECT_FLOAT_EQ(16,h[3]);
}

TEST(AgradFwdMatrixProd,ffv_matrix_3rdDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_ffv vv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;
   vv(0,0).val_.d_ = 1.0;
   vv(0,1).val_.d_ = 1.0;
   vv(1,0).val_.d_ = 1.0;
   vv(1,1).val_.d_ = 1.0;

  fvar<fvar<var> > f = prod(vv);

  AVEC q = createAVEC(vv(0,0).val().val(),vv(0,1).val().val(),vv(1,0).val().val(),vv(1,1).val().val());
  VEC h;
  f.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(16,h[0]);
  EXPECT_FLOAT_EQ(14,h[1]);
  EXPECT_FLOAT_EQ(16,h[2]);
  EXPECT_FLOAT_EQ(14,h[3]);
}
