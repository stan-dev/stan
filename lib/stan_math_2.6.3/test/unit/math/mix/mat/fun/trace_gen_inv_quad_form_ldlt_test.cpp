#include <stan/math/prim/mat/fun/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/rev/mat/fun/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/rev/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/rev/scal/fun/is_nan.hpp>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_fv_matrix_fv1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(A(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-51.64,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(-437.76, grads[1]);
  EXPECT_FLOAT_EQ(126, grads[2]);
  EXPECT_FLOAT_EQ(15.2, grads[3]);
  EXPECT_FLOAT_EQ(83.28, grads[4]);
  EXPECT_FLOAT_EQ(-12, grads[5]);
  EXPECT_FLOAT_EQ(15.2, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(145.2, grads[8]);
  EXPECT_FLOAT_EQ(18.4, grads[9]);
  EXPECT_FLOAT_EQ(-3.96, grads[10]);
  EXPECT_FLOAT_EQ(-13.8, grads[11]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_fv_matrix_fv2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(A(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-1.56, grads[0]);
  EXPECT_FLOAT_EQ(375.872, grads[1]);
  EXPECT_FLOAT_EQ(-47.2, grads[2]);
  EXPECT_FLOAT_EQ(-2.52, grads[3]);
  EXPECT_FLOAT_EQ(-136.176, grads[4]);
  EXPECT_FLOAT_EQ(13.8, grads[5]);
  EXPECT_FLOAT_EQ(-2.52, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(-56.32, grads[8]);
  EXPECT_FLOAT_EQ(-3.84, grads[9]);
  EXPECT_FLOAT_EQ(9.552, grads[10]);
  EXPECT_FLOAT_EQ(16.08, grads[11]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_fv_matrix_d1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(A(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(-437.76, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(83.28, grads[3]);
  EXPECT_FLOAT_EQ(15.2, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(18.4, grads[6]);
  EXPECT_FLOAT_EQ(-3.96, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_fv_matrix_d2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(A(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val());

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-6.76, grads[0]);
  EXPECT_FLOAT_EQ(592.83197, grads[1]);
  EXPECT_FLOAT_EQ(-8.32, grads[2]);
  EXPECT_FLOAT_EQ(-211.056, grads[3]);
  EXPECT_FLOAT_EQ(-8.32, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-10.24, grads[6]);
  EXPECT_FLOAT_EQ(14.712, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_d_matrix_fv1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(306.8,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(126, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(-12, grads[3]);
  EXPECT_FLOAT_EQ(15.2, grads[4]);
  EXPECT_FLOAT_EQ(145.2, grads[5]);
  EXPECT_FLOAT_EQ(18.4, grads[6]);
  EXPECT_FLOAT_EQ(-13.8, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_d_matrix_fv2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(D(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(5.2, grads[0]);
  EXPECT_FLOAT_EQ(44, grads[1]);
  EXPECT_FLOAT_EQ(5.8, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
  EXPECT_FLOAT_EQ(5.8, grads[4]);
  EXPECT_FLOAT_EQ(48.8, grads[5]);
  EXPECT_FLOAT_EQ(6.4, grads[6]);
  EXPECT_FLOAT_EQ(-10.2, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fv_matrix_fv1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(A(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-113.04,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-437.76, grads[0]);
  EXPECT_FLOAT_EQ(126, grads[1]);
  EXPECT_FLOAT_EQ(83.28, grads[2]);
  EXPECT_FLOAT_EQ(-12, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(145.2, grads[5]);
  EXPECT_FLOAT_EQ(-3.96, grads[6]);
  EXPECT_FLOAT_EQ(-13.8, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fv_matrix_fv2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
    vars.push_back(A(i).val_);
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(416.832, grads[0]);
  EXPECT_FLOAT_EQ(-60, grads[1]);
  EXPECT_FLOAT_EQ(-143.856, grads[2]);
  EXPECT_FLOAT_EQ(15, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-69.12, grads[5]);
  EXPECT_FLOAT_EQ(9.912, grads[6]);
  EXPECT_FLOAT_EQ(17.28, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_d_matrix_d1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    vars.push_back(D(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(15.2, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(18.4, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_fv_matrix_d_matrix_d2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    vars.push_back(D(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val());

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fv_matrix_d1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_ = 1.0;
    vars.push_back(A(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-437.76, grads[0]);
  EXPECT_FLOAT_EQ(83.28, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-3.96, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fv_matrix_d2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_ = 1.0;
    vars.push_back(A(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val());

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(633.792, grads[0]);
  EXPECT_FLOAT_EQ(-218.736, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(15.072, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_fv1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).d_ = 1.0;
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val());

  std::vector<double> grads;
  I.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(126, grads[0]);
  EXPECT_FLOAT_EQ(-12, grads[1]);
  EXPECT_FLOAT_EQ(145.2, grads[2]);
  EXPECT_FLOAT_EQ(-13.8, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_fv2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fv B(2,2);
  std::vector<var> vars;
  fvar<var> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).d_ = 1.0;
    vars.push_back(B(i).val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val());

  std::vector<double> grads;
  I.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(31.2, grads[0]);
  EXPECT_FLOAT_EQ(-7.8, grads[1]);
  EXPECT_FLOAT_EQ(36, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,exceptions_fv) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;

  matrix_fv fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<var>,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<var>,-1,-1> fv2;
  stan::math::LDLT_factor<double,-1,-1> fd1;
  stan::math::LDLT_factor<double,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv1, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv1, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv2, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv2, rvf1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvf1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvf1), 
               std::invalid_argument);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_ffv1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-51.64,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(-437.76, grads[1]);
  EXPECT_FLOAT_EQ(126, grads[2]);
  EXPECT_FLOAT_EQ(15.2, grads[3]);
  EXPECT_FLOAT_EQ(83.28, grads[4]);
  EXPECT_FLOAT_EQ(-12, grads[5]);
  EXPECT_FLOAT_EQ(15.2, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(145.2, grads[8]);
  EXPECT_FLOAT_EQ(18.4, grads[9]);
  EXPECT_FLOAT_EQ(-3.96, grads[10]);
  EXPECT_FLOAT_EQ(-13.8, grads[11]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_ffv2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-1.56, grads[0]);
  EXPECT_FLOAT_EQ(375.872, grads[1]);
  EXPECT_FLOAT_EQ(-47.2, grads[2]);
  EXPECT_FLOAT_EQ(-2.52, grads[3]);
  EXPECT_FLOAT_EQ(-136.176, grads[4]);
  EXPECT_FLOAT_EQ(13.8, grads[5]);
  EXPECT_FLOAT_EQ(-2.52, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(-56.32, grads[8]);
  EXPECT_FLOAT_EQ(-3.84, grads[9]);
  EXPECT_FLOAT_EQ(9.552, grads[10]);
  EXPECT_FLOAT_EQ(16.08, grads[11]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_ffv3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    A(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-1.56, grads[0]);
  EXPECT_FLOAT_EQ(375.872, grads[1]);
  EXPECT_FLOAT_EQ(-47.2, grads[2]);
  EXPECT_FLOAT_EQ(-2.52, grads[3]);
  EXPECT_FLOAT_EQ(-136.176, grads[4]);
  EXPECT_FLOAT_EQ(13.8, grads[5]);
  EXPECT_FLOAT_EQ(-2.52, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(-56.32, grads[8]);
  EXPECT_FLOAT_EQ(-3.84, grads[9]);
  EXPECT_FLOAT_EQ(9.552, grads[10]);
  EXPECT_FLOAT_EQ(16.08, grads[11]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_ffv4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    A(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(3.072, grads[0]);
  EXPECT_FLOAT_EQ(-620.82562, grads[1]);
  EXPECT_FLOAT_EQ(59.84, grads[2]);
  EXPECT_FLOAT_EQ(4.224, grads[3]);
  EXPECT_FLOAT_EQ(245.1008, grads[4]);
  EXPECT_FLOAT_EQ(-14.96, grads[5]);
  EXPECT_FLOAT_EQ(4.224, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
  EXPECT_FLOAT_EQ(70.784, grads[8]);
  EXPECT_FLOAT_EQ(5.808, grads[9]);
  EXPECT_FLOAT_EQ(-22.4736, grads[10]);
  EXPECT_FLOAT_EQ(-17.696, grads[11]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_d1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(-437.76, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(83.28, grads[3]);
  EXPECT_FLOAT_EQ(15.2, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(18.4, grads[6]);
  EXPECT_FLOAT_EQ(-3.96, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_d2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-6.76, grads[0]);
  EXPECT_FLOAT_EQ(592.83197, grads[1]);
  EXPECT_FLOAT_EQ(-8.32, grads[2]);
  EXPECT_FLOAT_EQ(-211.056, grads[3]);
  EXPECT_FLOAT_EQ(-8.32, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-10.24, grads[6]);
  EXPECT_FLOAT_EQ(14.712, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_d3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    A(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-6.76, grads[0]);
  EXPECT_FLOAT_EQ(592.83197, grads[1]);
  EXPECT_FLOAT_EQ(-8.32, grads[2]);
  EXPECT_FLOAT_EQ(-211.056, grads[3]);
  EXPECT_FLOAT_EQ(-8.32, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-10.24, grads[6]);
  EXPECT_FLOAT_EQ(14.712, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_ffv_matrix_d4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    A(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-297.04,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(8.112, grads[0]);
  EXPECT_FLOAT_EQ(-1100.5696, grads[1]);
  EXPECT_FLOAT_EQ(9.984, grads[2]);
  EXPECT_FLOAT_EQ(451.0528, grads[3]);
  EXPECT_FLOAT_EQ(9.984, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(12.288, grads[6]);
  EXPECT_FLOAT_EQ(-43.9776, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_ffv1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(306.8,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(126, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(-12, grads[3]);
  EXPECT_FLOAT_EQ(15.2, grads[4]);
  EXPECT_FLOAT_EQ(145.2, grads[5]);
  EXPECT_FLOAT_EQ(18.4, grads[6]);
  EXPECT_FLOAT_EQ(-13.8, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_ffv2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(5.2, grads[0]);
  EXPECT_FLOAT_EQ(44, grads[1]);
  EXPECT_FLOAT_EQ(5.8, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
  EXPECT_FLOAT_EQ(5.8, grads[4]);
  EXPECT_FLOAT_EQ(48.8, grads[5]);
  EXPECT_FLOAT_EQ(6.4, grads[6]);
  EXPECT_FLOAT_EQ(-10.2, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_ffv3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(5.2, grads[0]);
  EXPECT_FLOAT_EQ(44, grads[1]);
  EXPECT_FLOAT_EQ(5.8, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
  EXPECT_FLOAT_EQ(5.8, grads[4]);
  EXPECT_FLOAT_EQ(48.8, grads[5]);
  EXPECT_FLOAT_EQ(6.4, grads[6]);
  EXPECT_FLOAT_EQ(-10.2, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_ffv4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(1.2, grads[0]);
  EXPECT_FLOAT_EQ(6.4, grads[1]);
  EXPECT_FLOAT_EQ(1.2, grads[2]);
  EXPECT_FLOAT_EQ(-1.6, grads[3]);
  EXPECT_FLOAT_EQ(1.2, grads[4]);
  EXPECT_FLOAT_EQ(6.4, grads[5]);
  EXPECT_FLOAT_EQ(1.2, grads[6]);
  EXPECT_FLOAT_EQ(-1.6, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_ffv1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-113.04,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-437.76, grads[0]);
  EXPECT_FLOAT_EQ(126, grads[1]);
  EXPECT_FLOAT_EQ(83.28, grads[2]);
  EXPECT_FLOAT_EQ(-12, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(145.2, grads[5]);
  EXPECT_FLOAT_EQ(-3.96, grads[6]);
  EXPECT_FLOAT_EQ(-13.8, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_ffv2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(416.832, grads[0]);
  EXPECT_FLOAT_EQ(-60, grads[1]);
  EXPECT_FLOAT_EQ(-143.856, grads[2]);
  EXPECT_FLOAT_EQ(15, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-69.12, grads[5]);
  EXPECT_FLOAT_EQ(9.912, grads[6]);
  EXPECT_FLOAT_EQ(17.28, grads[7]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_ffv3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    A(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(416.832, grads[0]);
  EXPECT_FLOAT_EQ(-60, grads[1]);
  EXPECT_FLOAT_EQ(-143.856, grads[2]);
  EXPECT_FLOAT_EQ(15, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-69.12, grads[5]);
  EXPECT_FLOAT_EQ(9.912, grads[6]);
  EXPECT_FLOAT_EQ(17.28, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_ffv4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
    A(i).val_.d_ = 1.0;
    B(i).val_.d_ = 1.0;
    vars.push_back(A(i).val_.val_);
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-698.6496, grads[0]);
  EXPECT_FLOAT_EQ(72, grads[1]);
  EXPECT_FLOAT_EQ(271.85281, grads[2]);
  EXPECT_FLOAT_EQ(-18, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(82.944, grads[5]);
  EXPECT_FLOAT_EQ(-24.2976, grads[6]);
  EXPECT_FLOAT_EQ(-20.736, grads[7]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_d1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(12.6, grads[0]);
  EXPECT_FLOAT_EQ(15.2, grads[1]);
  EXPECT_FLOAT_EQ(15.2, grads[2]);
  EXPECT_FLOAT_EQ(18.4, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_d2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    vars.push_back(D(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_d3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_ffv_matrix_d_matrix_d4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffv D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    D(i).val_.d_ = 1.0;
    vars.push_back(D(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(61.4,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_d1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-437.76, grads[0]);
  EXPECT_FLOAT_EQ(83.28, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-3.96, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_d2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(633.792, grads[0]);
  EXPECT_FLOAT_EQ(-218.736, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(15.072, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_d3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    A(i).val_.d_ = 1.0;
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(633.792, grads[0]);
  EXPECT_FLOAT_EQ(-218.736, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(15.072, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffv_matrix_d4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffv A(2,2);
  stan::math::matrix_d B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    A(i).val_.d_ = 1.0;
    vars.push_back(A(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-358.44,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(-1219.3536, grads[0]);
  EXPECT_FLOAT_EQ(491.8848, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-46.7616, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_ffv1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).d_.val_ = 1.0;
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(126, grads[0]);
  EXPECT_FLOAT_EQ(-12, grads[1]);
  EXPECT_FLOAT_EQ(145.2, grads[2]);
  EXPECT_FLOAT_EQ(-13.8, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_ffv2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).d_.val_ = 1.0;
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.val_.grad(vars,grads);
  EXPECT_FLOAT_EQ(31.2, grads[0]);
  EXPECT_FLOAT_EQ(-7.8, grads[1]);
  EXPECT_FLOAT_EQ(36, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_ffv3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).val_.d_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val_.val());

  std::vector<double> grads;
  I.val_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(31.2, grads[0]);
  EXPECT_FLOAT_EQ(-7.8, grads[1]);
  EXPECT_FLOAT_EQ(36, grads[2]);
  EXPECT_FLOAT_EQ(-9, grads[3]);
}
TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_ffv4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffv B(2,2);
  std::vector<var> vars;
  fvar<fvar<var> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    B(i).val_.d_ = 1.0;
    B(i).d_.val_ = 1.0;
    vars.push_back(B(i).val_.val_);
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_.val());
  EXPECT_FLOAT_EQ(245.4,I.d_.val_.val());

  std::vector<double> grads;
  I.d_.d_.grad(vars,grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixTraceGenInvQuadFormLDLT,exceptions_ffv) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_ffv rvf1(3), rvf2(4);
  vector_ffv vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> fv2;
  stan::math::LDLT_factor<double,-1,-1> fd1;
  stan::math::LDLT_factor<double,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv1, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv1, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvd2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvf2), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv2, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv2, rvf1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv2_, fv1, rvf1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvd1), 
               std::invalid_argument);
  EXPECT_THROW(trace_gen_inv_quad_form_ldlt(fv1_, fv2, rvf1), 
               std::invalid_argument);
}
