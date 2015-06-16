#include <stan/math/prim/mat/fun/exp.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradMixMatrixExp, fv_matrix_1stDeriv) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;
  int i,j;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, fv_matrix_2ndDeriv) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, fv_vector_1stDeriv) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, fv_vector_2ndDeriv) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, fv_rowvector_1stDeriv) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, fv_rowvector_2ndDeriv) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_matrix_1stDeriv) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;
  int i,j;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_matrix_2ndDeriv_1) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_matrix_2ndDeriv_2) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_matrix_3rdDeriv) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1.0),h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_vector_1stDeriv) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_vector_2ndDeriv_1) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_vector_2ndDeriv_2) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_vector_3rdDeriv) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  vector_d expected_output(4);
  vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1.0),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_rowvector_1stDeriv) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val().val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_rowvector_2ndDeriv_1) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_rowvector_2ndDeriv_2) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixExp, ffv_rowvector_3rdDeriv) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  mv << a,b,c,d;
  output = exp(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(std::exp(1.0),h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
