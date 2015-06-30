#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/fun/mdivide_right_spd.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixMdivideRightSPD,matrix_fd_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_right_spd;

  matrix_fd Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_);
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.1388889,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.10416666,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.80555558,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.60416669,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_fd_row_vector_fd) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_spd;

  matrix_fd Av(2,2);
  row_vector_fd Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_);
  EXPECT_FLOAT_EQ(-0.1388889,I(0).d_);
  EXPECT_FLOAT_EQ(-0.10416666,I(1).d_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_fd_matrix_d) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_fd Av(2,2);
  matrix_d Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_);
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.47222221,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.35416666,I(0,1).d_);
  EXPECT_FLOAT_EQ(-1.1388888,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.85416669,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_fd_row_vector_d) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_fd Av(2,2);
  row_vector_d Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_);
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_);
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_d_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_);
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_);
  EXPECT_FLOAT_EQ(0.33333334,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_d_row_vector_fd) {
  using stan::math::matrix_d;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_fd Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_);
}

TEST(AgradFwdMatrixMdivideRightSPD,fd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_spd;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  row_vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right_spd(fd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vf1, fd2), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideRightSPD,matrix_ffd_matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_right_spd;

  matrix_ffd Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.1388889,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.10416666,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.80555558,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.60416669,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_ffd_row_vector_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_spd;

  matrix_ffd Av(2,2);
  row_vector_ffd Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.1388889,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.10416666,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_ffd_matrix_d) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_ffd Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.47222221,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.35416666,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-1.1388888,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.85416669,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_ffd_row_vector_d) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_ffd Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_d_matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightSPD,matrix_d_row_vector_ffd) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_ffd Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightSPD,ffd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_spd;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  row_vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right_spd(fd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(fv2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(rvf2, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_spd(vf1, fd2), std::invalid_argument);
}
