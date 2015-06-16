#include <stan/math/prim/mat/fun/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/rev/mat/fun/trace_inv_quad_form_ldlt.hpp>
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


TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_matrix_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Ad);
  EXPECT_FLOAT_EQ(9.0,I.val_.val());
  EXPECT_FLOAT_EQ(2.0,I.d_.val());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val());
  EXPECT_FLOAT_EQ(5.04,I.d_.val());
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.04, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.32, grads[2]);
  EXPECT_FLOAT_EQ(-1.64, grads[3]);
  EXPECT_FLOAT_EQ(-0.4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(1.6, grads[6]);
  EXPECT_FLOAT_EQ(2, grads[7]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_matrix_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.448, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-3.504, grads[2]);
  EXPECT_FLOAT_EQ(0.848, grads[3]);
  EXPECT_FLOAT_EQ(2.24, grads[4]);
  EXPECT_FLOAT_EQ(1.6, grads[5]);
  EXPECT_FLOAT_EQ(-0.56, grads[6]);
  EXPECT_FLOAT_EQ(-0.4, grads[7]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val());
  EXPECT_FLOAT_EQ(-1.36,I.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.04, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.32, grads[2]);
  EXPECT_FLOAT_EQ(-1.64, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.192, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(2.416, grads[2]);
  EXPECT_FLOAT_EQ(-0.592, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_fv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val());
  EXPECT_FLOAT_EQ(6.4,I.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.6, grads[2]);
  EXPECT_FLOAT_EQ(2, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_fv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(3.2, grads[1]);
  EXPECT_FLOAT_EQ(-0.8, grads[2]);
  EXPECT_FLOAT_EQ(-0.8, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_vector_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_fv Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val());
  EXPECT_FLOAT_EQ(3,I.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(2, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_vector_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_fv Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.6, grads[4]);
  EXPECT_FLOAT_EQ(-0.4, grads[5]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_vector_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val());
  EXPECT_FLOAT_EQ(-1,I.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_fv_vector_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_fv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_fv Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val());
  EXPECT_FLOAT_EQ(4,I.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(2, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_fv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_fv Av(2);
  fvar<var> I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(-0.8, grads[1]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,fv_exceptions) {
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
  stan::math::LDLT_factor<double ,-1,-1> fd1;
  stan::math::LDLT_factor<double ,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, vf1), std::invalid_argument);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Ad);
  EXPECT_FLOAT_EQ(9.0,I.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0,I.d_.val_.val());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val_.val());
  EXPECT_FLOAT_EQ(5.04,I.d_.val_.val());
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.04, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.32, grads[2]);
  EXPECT_FLOAT_EQ(-1.64, grads[3]);
  EXPECT_FLOAT_EQ(-0.4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(1.6, grads[6]);
  EXPECT_FLOAT_EQ(2, grads[7]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.448, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-3.504, grads[2]);
  EXPECT_FLOAT_EQ(0.848, grads[3]);
  EXPECT_FLOAT_EQ(2.24, grads[4]);
  EXPECT_FLOAT_EQ(1.6, grads[5]);
  EXPECT_FLOAT_EQ(-0.56, grads[6]);
  EXPECT_FLOAT_EQ(-0.4, grads[7]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;
  Av(0,0).val_.d_ = 2.0;
  Av(0,1).val_.d_ = 2.0;
  Av(1,0).val_.d_ = 2.0;
  Av(1,1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.448, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-3.504, grads[2]);
  EXPECT_FLOAT_EQ(0.848, grads[3]);
  EXPECT_FLOAT_EQ(2.24, grads[4]);
  EXPECT_FLOAT_EQ(1.6, grads[5]);
  EXPECT_FLOAT_EQ(-0.56, grads[6]);
  EXPECT_FLOAT_EQ(-0.4, grads[7]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;
  Av(0,0).val_.d_ = 2.0;
  Av(0,1).val_.d_ = 2.0;
  Av(1,0).val_.d_ = 2.0;
  Av(1,1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  
  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-4.3263998, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(6.0991998, grads[2]);
  EXPECT_FLOAT_EQ(-1.2544, grads[3]);
  EXPECT_FLOAT_EQ(-2.688, grads[4]);
  EXPECT_FLOAT_EQ(-1.92, grads[5]);
  EXPECT_FLOAT_EQ(0.672, grads[6]);
  EXPECT_FLOAT_EQ(0.48, grads[7]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-1.36,I.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.04, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.32, grads[2]);
  EXPECT_FLOAT_EQ(-1.64, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.192, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(2.416, grads[2]);
  EXPECT_FLOAT_EQ(-0.592, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.192, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(2.416, grads[2]);
  EXPECT_FLOAT_EQ(-0.592, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_matrix_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.5104, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-2.0288, grads[2]);
  EXPECT_FLOAT_EQ(0.6016, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_ffv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(10.6,I.val_.val_.val());
  EXPECT_FLOAT_EQ(6.4,I.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.6, grads[2]);
  EXPECT_FLOAT_EQ(2, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_ffv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(3.2, grads[1]);
  EXPECT_FLOAT_EQ(-0.8, grads[2]);
  EXPECT_FLOAT_EQ(-0.8, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_ffv_3) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;
  Av(0,0).val_.d_ = 2.0;
  Av(0,1).val_.d_ = 2.0;
  Av(1,0).val_.d_ = 2.0;
  Av(1,1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(3.2, grads[1]);
  EXPECT_FLOAT_EQ(-0.8, grads[2]);
  EXPECT_FLOAT_EQ(-0.8, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_matrix_ffv_4) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_.val_ = 2.0;
  Av(0,1).d_.val_ = 2.0;
  Av(1,0).d_.val_ = 2.0;
  Av(1,1).d_.val_ = 2.0;
  Av(0,0).val_.d_ = 2.0;
  Av(0,1).val_.d_ = 2.0;
  Av(1,0).val_.d_ = 2.0;
  Av(1,1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val_.val());
  EXPECT_FLOAT_EQ(3,I.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(2, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.6, grads[4]);
  EXPECT_FLOAT_EQ(-0.4, grads[5]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.6, grads[4]);
  EXPECT_FLOAT_EQ(-0.4, grads[5]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.64, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.16, grads[2]);
  EXPECT_FLOAT_EQ(-0.08, grads[3]);
  EXPECT_FLOAT_EQ(-1.92, grads[4]);
  EXPECT_FLOAT_EQ(0.48, grads[5]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val_.val());
  EXPECT_FLOAT_EQ(-1,I.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.4, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_ffv_vector_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-3.2, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.12, grads[2]);
  EXPECT_FLOAT_EQ(-0.08, grads[3]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_ffv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(2,I.val_.val_.val());
  EXPECT_FLOAT_EQ(4,I.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(2, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_ffv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(-0.8, grads[1]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_ffv_3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(3.2, grads[0]);
  EXPECT_FLOAT_EQ(-0.8, grads[1]);
}
TEST(AgradMixMatrixTraceInvQuadFormLDLT,matrix_d_vector_ffv_4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  fvar<fvar<var> > I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = trace_inv_quad_form_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixTraceInvQuadFormLDLT,ffv_exceptions) {
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
  stan::math::LDLT_factor<double ,-1,-1> fd1;
  stan::math::LDLT_factor<double ,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(trace_inv_quad_form_ldlt(fd2, vf1), std::invalid_argument);
}
