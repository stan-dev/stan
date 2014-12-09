#include <stan/math/matrix/mdivide_right_ldlt.hpp>
#include <stan/agrad/fwd/matrix/mdivide_left_ldlt.hpp>
#include <stan/agrad/rev/matrix/mdivide_left_ldlt.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>


using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fd_matrix_fd) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fd Ad(2,2);
  stan::agrad::matrix_fd Av(2,2);
  stan::agrad::matrix_fd I;

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

  I = mdivide_right_ldlt(Ad,ldlt_Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_);
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_);
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_);

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.8,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.2,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.48,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.12,I(1,1).d_);
}


TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fd_matrix_d) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fd Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_fd I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.8,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.2,I(0,1).d_);
  EXPECT_FLOAT_EQ(-2.08,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.52,I(1,1).d_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_fd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_fd Av(2,2);
  stan::agrad::matrix_fd I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_);
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.4,I(0,1).d_);
  EXPECT_FLOAT_EQ(1.6,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fd_row_vector_fd) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fd Ad(2,2);
  stan::agrad::row_vector_fd Av(2);
  stan::agrad::row_vector_fd I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_);
  EXPECT_FLOAT_EQ(0.0,I(1).val_);
  EXPECT_FLOAT_EQ(0.8,I(0).d_);
  EXPECT_FLOAT_EQ(-0.2,I(1).d_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fd_row_vector_d) {
  stan::math::LDLT_factor<fvar<double>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fd Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_fd I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_);
  EXPECT_FLOAT_EQ(0,I(1).val_);
  EXPECT_FLOAT_EQ(-0.8,I(0).d_);
  EXPECT_FLOAT_EQ(0.2,I(1).d_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_fd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_fd Av(2);
  stan::agrad::row_vector_fd I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_);
  EXPECT_FLOAT_EQ(0.0,I(1).val_);
  EXPECT_FLOAT_EQ(1.6,I(0).d_);
  EXPECT_FLOAT_EQ(-0.4,I(1).d_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,fd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;

  matrix_fd fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_fd rvf1(3), rvf2(4);
  row_vector_fd vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<double>,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<double>,-1,-1> fv2;
  stan::math::LDLT_factor<double,-1,-1> fd1;
  stan::math::LDLT_factor<double,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fd2), std::domain_error);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffd_matrix_ffd) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffd Ad(2,2);
  stan::agrad::matrix_ffd Av(2,2);
  stan::agrad::matrix_ffd I;

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

  I = mdivide_right_ldlt(Ad,ldlt_Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_.val_);

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.8,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.2,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.48,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.12,I(1,1).d_.val_);
}


TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffd_matrix_d) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffd Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_ffd I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.8,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.2,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-2.08,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.52,I(1,1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_ffd) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_ffd Av(2,2);
  stan::agrad::matrix_ffd I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.4,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(1.6,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffd_row_vector_ffd) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffd Ad(2,2);
  stan::agrad::row_vector_ffd Av(2);
  stan::agrad::row_vector_ffd I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.8,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.2,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffd_row_vector_d) {
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffd Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_ffd I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.8,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.2,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_ffd) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_ffd Av(2);
  stan::agrad::row_vector_ffd I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(1.6,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.4,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightLDLT,ffd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;

  matrix_ffd fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_ffd rvf1(3), rvf2(4);
  row_vector_ffd vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<fvar<double> >,-1,-1> fv2;
  stan::math::LDLT_factor<double ,-1,-1> fd1;
  stan::math::LDLT_factor<double ,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fd2), std::domain_error);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_matrix_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::agrad::matrix_fv Av(2,2);
  stan::agrad::matrix_fv I;

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

  I = mdivide_right_ldlt(Ad,ldlt_Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_.val());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.8,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.2,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.48,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.12,I(1,1).d_.val());
  
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

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(-0.6, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_matrix_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::agrad::matrix_fv Av(2,2);
  stan::agrad::matrix_fv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  
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

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_fv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.2,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-2.08,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.52,I(1,1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_fv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_fv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_fv Av(2,2);
  stan::agrad::matrix_fv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.4,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.6,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(-0.6, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_fv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_fv Av(2,2);
  stan::agrad::matrix_fv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_row_vector_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::agrad::row_vector_fv Av(2);
  stan::agrad::row_vector_fv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(0.8,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(-0.6, grads[5]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_row_vector_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::agrad::row_vector_fv Av(2);
  stan::agrad::row_vector_fv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_row_vector_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val());
  EXPECT_FLOAT_EQ(0,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0).d_.val());
  EXPECT_FLOAT_EQ(0.2,I(1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_fv_row_vector_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::agrad::matrix_fv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_fv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_fv Av(2);
  stan::agrad::row_vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(-0.6, grads[1]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_fv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_fv Av(2);
  stan::agrad::row_vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,fv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::vector_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;

  matrix_fv fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_fv rvf1(3), rvf2(4);
  row_vector_fv vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<var>,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<var>,-1,-1> fv2;
  stan::math::LDLT_factor<double ,-1,-1> fd1;
  stan::math::LDLT_factor<double ,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fd2), std::domain_error);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Ad,ldlt_Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_.val_.val());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.48,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.12,I(1,1).d_.val_.val());
  
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

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(-0.6, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  
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

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  
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

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  
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

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.232, grads[2]);
  EXPECT_FLOAT_EQ(0.208, grads[3]);
  EXPECT_FLOAT_EQ(0.768, grads[4]);
  EXPECT_FLOAT_EQ(-0.192, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.2,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-2.08,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.52,I(1,1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_matrix_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::agrad::matrix_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-3.136, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.616, grads[2]);
  EXPECT_FLOAT_EQ(-0.208, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_ffv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(2.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(1.6,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(-0.6, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_ffv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_ffv_3) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_matrix_ffv_4) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::matrix_ffv Av(2,2);
  stan::agrad::matrix_ffv I;

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
  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(-0.6, grads[5]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.48, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_FLOAT_EQ(-0.12, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0.16, grads[5]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.6, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.232, grads[2]);
  EXPECT_FLOAT_EQ(0.208, grads[3]);
  EXPECT_FLOAT_EQ(0.768, grads[4]);
  EXPECT_FLOAT_EQ(-0.192, grads[5]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.2,I(1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.6, grads[2]);
  EXPECT_NEAR(0, grads[3], 1E-12);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.76, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.92, grads[2]);
  EXPECT_FLOAT_EQ(0.12, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_ffv_row_vector_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::agrad::matrix_ffv Ad(2,2);
  stan::math::row_vector_d Av(2);
  stan::agrad::row_vector_ffv I;

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

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-3.136, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.616, grads[2]);
  EXPECT_FLOAT_EQ(-0.208, grads[3]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_ffv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);
  EXPECT_FLOAT_EQ(1,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(-0.6, grads[1]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_ffv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_ffv_3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideRightLDLT,matrix_d_row_vector_ffv_4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::agrad::row_vector_ffv Av(2);
  stan::agrad::row_vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_right_ldlt(Av,ldlt_Ad);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideRightLDLT,ffv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;

  matrix_ffv fv1_(3,3), fv2_(4,4);
  fv1_ << 1,2,3,4,5,6,7,8,9;
  fv2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_ffv rvf1(3), rvf2(4);
  row_vector_ffv vf1(3), vf2(4);
  matrix_d fd1_(3,3), fd2_(4,4);
  fd1_ << 1,2,3,4,5,6,7,8,9;
  fd2_ << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> fv1;
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> fv2;
  stan::math::LDLT_factor<double ,-1,-1> fd1;
  stan::math::LDLT_factor<double ,-1,-1> fd2;
  fv1.compute(fv1_);
  fv2.compute(fv2_);
  fd1.compute(fd1_);
  fd2.compute(fd2_);

  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fd2_, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(fv2_, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(rvf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_ldlt(vf1, fd2), std::domain_error);
}
