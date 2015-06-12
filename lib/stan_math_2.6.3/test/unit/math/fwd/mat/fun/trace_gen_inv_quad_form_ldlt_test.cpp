#include <stan/math/prim/mat/fun/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>

using stan::math::fvar;

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_fd_matrix_fd_matrix_fd) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_A;
  stan::math::matrix_fd D(2,2);
  stan::math::matrix_fd A(2,2);
  stan::math::matrix_fd B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(-51.64,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_fd_matrix_fd_matrix_d) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_A;
  stan::math::matrix_fd D(2,2);
  stan::math::matrix_fd A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    A(i).d_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(-297.04,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_fd_matrix_d_matrix_fd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fd D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fd B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_ = 1.0;
    B(i).d_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(306.8,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fd_matrix_fd) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fd A(2,2);
  stan::math::matrix_fd B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_ = 1.0;
    B(i).d_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(-113.04,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_fd_matrix_d_matrix_d) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_fd D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    D(i).d_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(61.4,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_fd_matrix_d) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_fd A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    A(i).d_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(-358.44,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_fd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_fd B(2,2);
  fvar<double> I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    B(i).d_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_);
  EXPECT_FLOAT_EQ(245.4,I.d_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,exceptions_fd) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;

  matrix_fd fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<double>,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<double>,-1,-1> fv2;
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

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_ffd_matrix_ffd_matrix_ffd) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_A;
  stan::math::matrix_ffd D(2,2);
  stan::math::matrix_ffd A(2,2);
  stan::math::matrix_ffd B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(-51.64,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_ffd_matrix_ffd_matrix_d) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_A;
  stan::math::matrix_ffd D(2,2);
  stan::math::matrix_ffd A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    A(i).d_.val_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(-297.04,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_ffd_matrix_d_matrix_ffd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffd D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffd B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    D(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(306.8,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffd_matrix_ffd) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffd A(2,2);
  stan::math::matrix_ffd B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++) {
    A(i).d_.val_ = 1.0;
    B(i).d_.val_ = 1.0;
  }

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(-113.04,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_ffd_matrix_d_matrix_d) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_ffd D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    D(i).d_.val_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(61.4,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_ffd_matrix_d) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_ffd A(2,2);
  stan::math::matrix_d B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    A(i).d_.val_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(-358.44,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,matrix_d_matrix_d_matrix_ffd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  stan::math::matrix_d D(2,2);
  stan::math::matrix_d A(2,2);
  stan::math::matrix_ffd B(2,2);
  fvar<fvar<double> > I;

  A << 2,3,3,7;
  B << 5,6,7,8;
  D << 9,10,11,12;
  for (int i = 0; i < 4; i++)
    B(i).d_.val_ = 1.0;

  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  I = trace_gen_inv_quad_form_ldlt(D,ldlt_A,B);
  EXPECT_FLOAT_EQ(653.4,I.val_.val_);
  EXPECT_FLOAT_EQ(245.4,I.d_.val_);
}

TEST(AgradFwdMatrixTraceGenInvQuadFormLDLT,exceptions_ffd) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;

  matrix_ffd fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> fv2;
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
