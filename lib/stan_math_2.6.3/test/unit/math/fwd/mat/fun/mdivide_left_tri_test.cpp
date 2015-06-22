#include <stan/math/prim/mat/fun/mdivide_left_tri.hpp>
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
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
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

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.11111111,I(0,0).d_);
  EXPECT_FLOAT_EQ(0,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.21527778,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.375,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_vector_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  vector_fd Ad(2);
  vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_);
  EXPECT_FLOAT_EQ(0.11111111,I(0).d_);
  EXPECT_FLOAT_EQ(-0.090277776,I(1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_d_lower) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_d Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.22222222,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.33333333,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.38194445,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.54166669,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_vector_d_lower) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  vector_d Ad(2);
  vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_);
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_);
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,1).d_);
  EXPECT_FLOAT_EQ(0.16666666,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.16666666,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fd_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fd Ad(2);
  vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_);
  EXPECT_FLOAT_EQ(0.16666666,I(1).d_);
}

TEST(AgradFwdMatrixMdivideLeftTri,fd_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
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

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.11111111,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.21527778,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.375,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_vector_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  vector_ffd Ad(2);
  vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.11111111,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.090277776,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_d_lower) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.22222222,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.33333333,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.38194445,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.54166669,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_vector_d_lower) {
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  vector_d Ad(2);
  vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0.16666666,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.16666666,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffd_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffd Ad(2);
  vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.16666666,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideLeftTri,ffd_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_fd_upper) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 1.0, 
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

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.14583333,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_vector_fd_upper) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  vector_fd Ad(2);
  vector_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(-0.076388888,I(0).d_);
  EXPECT_FLOAT_EQ(0.0625,I(1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_d_upper) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_d Ad(2,2);
  matrix_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.39583334,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.5763889,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.3125,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_vector_d_upper) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  vector_d Ad(2);
  vector_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_);
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fd_upper) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fd Ad(2,2);
  matrix_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_);
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fd_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fd Ad(2);
  vector_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(0.25,I(0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_);
}

TEST(AgradFwdMatrixMdivideLeftTri,fd_exceptions_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_ffd_upper) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 1.0, 
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

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.14583333,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_vector_ffd_upper) {
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  vector_ffd Ad(2);
  vector_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.076388888,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.0625,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_d_upper) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.39583334,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.5763889,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.3125,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_vector_d_upper) {
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  vector_d Ad(2);
  vector_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffd_upper) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffd Ad(2,2);
  matrix_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffd_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffd Ad(2);
  vector_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideLeftTri,ffd_exceptions_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_upper) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_fd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  
  I = mdivide_left_tri<Eigen::Upper>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_);
  EXPECT_FLOAT_EQ(-0.083333333,I(0,1).val_);
  EXPECT_FLOAT_EQ(0,I(1,0).val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.034722224,I(0,1).d_);
  EXPECT_FLOAT_EQ(0,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_upper) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_ffd I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(-0.083333333,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(0,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.034722224,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd Av(2,2);
  matrix_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  
  I = mdivide_left_tri<Eigen::Lower>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_);
  EXPECT_FLOAT_EQ(0,I(0,1).val_);
  EXPECT_FLOAT_EQ(-0.083333333,I(1,0).val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_);
  EXPECT_FLOAT_EQ(0,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.034722224,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd Av(2,2);
  matrix_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.083333333,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.034722224,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val_);
}
