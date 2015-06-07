#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/prod.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixMatrixProd,fv_vector_1stDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,fv_vector_2ndDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixProd,fv_rowvector_1stDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,fv_rowvector_2ndDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,fv_matrix_1stDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,fv_matrix_2ndDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_vector_1stDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_vector_2ndDeriv_1) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_vector_2ndDeriv_2) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_vector_3rdDeriv) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixProd,ffv_rowvector_1stDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_rowvector_2ndDeriv_1) {
  using stan::math::prod;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_rowvector_2ndDeriv_2) {
  using stan::math::prod;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_rowvector_3rdDeriv) {
  using stan::math::prod;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_matrix_1stDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixProd,ffv_matrix_2ndDeriv_1) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixProd,ffv_matrix_2ndDeriv_2) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixProd,ffv_matrix_3rdDeriv) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
