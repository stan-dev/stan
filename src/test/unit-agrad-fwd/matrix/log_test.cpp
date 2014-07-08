#include <stan/math/matrix/log.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixLog, fd_matrix) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d expected_output(2,2);
  matrix_fd mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
   mv(0,0).d_ = 1.0;
   mv(1,0).d_ = 1.0;
   mv(0,1).d_ = 1.0;
   mv(1,1).d_ = 1.0;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_);
  EXPECT_FLOAT_EQ(1, output(0,0).d_);
  EXPECT_FLOAT_EQ(0.5, output(0,1).d_);
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(1,0).d_);
  EXPECT_FLOAT_EQ(0.25, output(1,1).d_);
}

TEST(AgradFwdMatrixLog, fd_vector) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d expected_output(4);
  vector_fd mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 1.0;
   mv(1).d_ = 1.0;
   mv(2).d_ = 1.0;
   mv(3).d_ = 1.0;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(0.5, output(1).d_);
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_);
  EXPECT_FLOAT_EQ(0.25, output(3).d_);
}

TEST(AgradFwdMatrixLog, fd_rowvector) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d expected_output(4);
  row_vector_fd mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 1.0;
   mv(1).d_ = 1.0;
   mv(2).d_ = 1.0;
   mv(3).d_ = 1.0;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(0.5, output(1).d_);
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_);
  EXPECT_FLOAT_EQ(0.25, output(3).d_);
}
TEST(AgradFwdMatrixLog, fv_matrix_1stDeriv) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;
  int i,j;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val());
  EXPECT_FLOAT_EQ(1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(1,1).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, fv_matrix_2ndDeriv) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, fv_vector_1stDeriv) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  vector_d expected_output(4);
  vector_fv mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, fv_vector_2ndDeriv) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  vector_fv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, fv_rowvector_1stDeriv) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, fv_rowvector_2ndDeriv) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffd_matrix) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0; 

  matrix_d expected_output(2,2);
  matrix_ffd mv(2,2), output;
  int i,j;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val());
  EXPECT_FLOAT_EQ(1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(1,1).d_.val());
}

TEST(AgradFwdMatrixLog, ffd_vector) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0; 

  vector_d expected_output(4);
  vector_ffd mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val());
}

TEST(AgradFwdMatrixLog, ffd_rowvector) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0; 

  row_vector_d expected_output(4);
  row_vector_ffd mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val());
}
TEST(AgradFwdMatrixLog, ffv_matrix_1stDeriv) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;
  int i,j;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val().val());
  EXPECT_FLOAT_EQ(1, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(0.5, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(0.25, output(1,1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_matrix_2ndDeriv_1) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_matrix_2ndDeriv_2) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_matrix_3rdDeriv) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  matrix_d expected_output(2,2);
  matrix_ffv mv(2,2), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, ffv_vector_1stDeriv) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  vector_d expected_output(4);
  vector_ffv mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val().val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val().val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_vector_2ndDeriv_1) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_vector_2ndDeriv_2) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_vector_3rdDeriv) {
  using stan::math::log;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, ffv_rowvector_1stDeriv) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  mv << a,b,c,d;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val().val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(1.0 / 3.0, output(2).d_.val().val());
  EXPECT_FLOAT_EQ(0.25, output(3).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixLog, ffv_rowvector_2ndDeriv_1) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, ffv_rowvector_2ndDeriv_2) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixLog, ffv_rowvector_3rdDeriv) {
  using stan::math::log;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  row_vector_d expected_output(4);
  row_vector_ffv mv(4), output;

  mv << a,b,c,d;
  output = log(mv);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
