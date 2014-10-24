#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
using stan::agrad::var;
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_fd_lower) {
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
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
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_ffd_lower) {
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_fv1_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.11111111,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.21527778,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.375,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_fv2_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_fv1_lower) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val());
  EXPECT_FLOAT_EQ(0.11111111,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.090277776,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_fv2_lower) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_d1_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.33333333,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.38194445,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.54166669,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_d2_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.33333333,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.38194445,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.54166669,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_d1_lower) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_d Ad(2);
  vector_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_d2_lower) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_d Ad(2);
  vector_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fv1_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fv2_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fv1_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fv2_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,fv_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.11111111,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.21527778,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.375,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_NEAR(0, grads[0], 1E-12);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.074074075, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.11111111,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.090277776,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.037037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.074074075, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.33333333,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.38194445,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.54166669,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.22222222,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.25694445,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.14814815, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.5,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv1_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.58333331,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.16666666,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv2_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv3_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv4_lower) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,ffv_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv fv1(3,3), fv2(4,4);
  row_vector_ffv rvf1(3), rvf2(4);
  vector_ffv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Lower>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_matrix_fd_upper) {
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
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
  using stan::agrad::vector_fd;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::math::mdivide_left_tri;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffd_matrix_ffd_upper) {
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
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
  using stan::agrad::vector_ffd;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::math::mdivide_left_tri;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_fv1_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.14583333,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.083333336, grads[0]);
  EXPECT_FLOAT_EQ(-0.41666666, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.10416666, grads[3]);
  EXPECT_FLOAT_EQ(0.33333333, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.083333333, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_fv2_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.076388888, grads[0]);
  EXPECT_FLOAT_EQ(0.15972222, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.038194444, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.034722224, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_fv1_upper) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

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
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.076388888,I(0).d_.val());
  EXPECT_FLOAT_EQ(0.0625,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.1388889, grads[0]);
  EXPECT_FLOAT_EQ(-0.25, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(0.33333333, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333333, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_fv2_upper) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.071759261, grads[0]);
  EXPECT_FLOAT_EQ(0.0625, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.03125, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.034722224, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_d1_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.39583334,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.5763889,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.3125,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.083333336, grads[0]);
  EXPECT_FLOAT_EQ(-0.41666666, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.10416666, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_matrix_d2_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.39583334,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.5763889,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.3125,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.15972222, grads[0]);
  EXPECT_FLOAT_EQ(0.24305555, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.017361112, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_d1_upper) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_d Ad(2);
  vector_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.1388889, grads[0]);
  EXPECT_FLOAT_EQ(-0.25, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv_vector_d2_upper) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  vector_d Ad(2);
  vector_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.1550926, grads[0]);
  EXPECT_FLOAT_EQ(0.14583333, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.010416667, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fv1_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333333, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.083333333, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_fv2_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fv1_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_fv2_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_fv Ad(2);
  vector_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,fv_exceptions_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.14583333,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.083333336, grads[0]);
  EXPECT_FLOAT_EQ(-0.41666666, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.10416666, grads[3]);
  EXPECT_FLOAT_EQ(0.33333333, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.083333333, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.076388888, grads[0]);
  EXPECT_FLOAT_EQ(0.15972222, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.038194444, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.034722222, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.076388888, grads[0]);
  EXPECT_FLOAT_EQ(0.15972222, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.038194444, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.034722222, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_ffv4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.09375, grads[0]);
  EXPECT_FLOAT_EQ(-0.11689815, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0.069733799, grads[3]);
  EXPECT_FLOAT_EQ(0.074074075, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0.054398149, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

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
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.076388888,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0625,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.1388889, grads[0]);
  EXPECT_FLOAT_EQ(-0.25, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(0.33333333, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333333, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.071759261, grads[0]);
  EXPECT_FLOAT_EQ(0.0625, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.03125, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.034722224, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.071759261, grads[0]);
  EXPECT_FLOAT_EQ(0.0625, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.03125, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.034722224, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_ffv4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.054398149, grads[0]);
  EXPECT_FLOAT_EQ(-0.03125, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0.0390625, grads[3]);
  EXPECT_FLOAT_EQ(0.074074075, grads[4]);
  EXPECT_FLOAT_EQ(0.054398149, grads[5]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.39583334,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.5763889,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.3125,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.083333336, grads[0]);
  EXPECT_FLOAT_EQ(-0.41666666, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.10416666, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.15972222, grads[0]);
  EXPECT_FLOAT_EQ(0.24305555, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.017361112, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.15972222, grads[0]);
  EXPECT_FLOAT_EQ(0.24305555, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.017361112, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_matrix_d4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.24652778, grads[0]);
  EXPECT_FLOAT_EQ(-0.21412037, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0.076678239, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.1388889, grads[0]);
  EXPECT_FLOAT_EQ(-0.25, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.1550926, grads[0]);
  EXPECT_FLOAT_EQ(0.14583333, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.010416667, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.1550926, grads[0]);
  EXPECT_FLOAT_EQ(0.14583333, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0.010416667, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv_vector_d4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  vector_d Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.20717593, grads[0]);
  EXPECT_FLOAT_EQ(-0.12847222, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0.046006944, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.25,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.41666666,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.25,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_matrix_ffv4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Ad(0,0).val_.d_ = 1.0;
  Ad(0,1).val_.d_ = 1.0;
  Ad(1,0).val_.d_ = 1.0;
  Ad(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv1_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_FLOAT_EQ(0.41666666,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[1]);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv2_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv3_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_d_vector_ffv4_upper) {
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_d Av(2,2);
  vector_ffv Ad(2);
  vector_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradFwdMatrixMdivideLeftTri,ffv_exceptions_upper) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv fv1(3,3), fv2(4,4);
  row_vector_ffv rvf1(3), rvf2(4);
  vector_ffv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left_tri<Eigen::Upper>(fd2, vf1), std::domain_error);
}

TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_upper) {
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_ffd;
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
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv1_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(-0.083333333,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.034722224,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.11111111, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv2_upper) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv1_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.083333333,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.034722224,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.11111111, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv2_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv3_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv4_upper) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 1.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Upper>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}


TEST(AgradFwdMatrixMdivideLeftTri,matrix_fd_lower) {
  using stan::agrad::matrix_fd;
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
  using stan::agrad::matrix_ffd;
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
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv1_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(-0.083333336,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.034722224,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.11111111, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_fv2_lower) {
  using stan::agrad::matrix_fv;
  using stan::math::mdivide_left_tri;

  matrix_fv Av(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv1_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);
  EXPECT_FLOAT_EQ(0.33333333,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.083333336,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.11111111,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.034722224,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.0625,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.11111111, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv2_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv3_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradFwdMatrixMdivideLeftTri,matrix_ffv4_lower) {
  using stan::agrad::matrix_ffv;
  using stan::math::mdivide_left_tri;

  matrix_ffv Av(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;

  I = mdivide_left_tri<Eigen::Lower>(Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.074074075, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

