#include <stan/math/prim/mat/fun/mdivide_right_tri.hpp>
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
TEST(AgradFwdMatrixMdivideRightTri,matrix_fd_matrix_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.076388888,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.0625,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.5486111,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_fd_row_vector_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_tri;

  matrix_fd Av(2,2);
  row_vector_fd Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(-0.0763888889,I(0).d_);
  EXPECT_FLOAT_EQ(0.0625,I(1).d_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_fd_matrix_d_lower) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0,0).d_);
  EXPECT_FLOAT_EQ(-0.1875,I(0,1).d_);
  EXPECT_FLOAT_EQ(-0.7986111,I(1,0).d_);
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_fd_row_vector_d_lower) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_tri;

  matrix_fd Av(2,2);
  row_vector_d Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_);
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_d_matrix_fd_lower) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0,0).val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_);
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_);
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_d_row_vector_fd_lower) {
  using stan::math::matrix_d;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_tri;

  matrix_d Av(2,2);
  row_vector_fd Ad(2);
  row_vector_fd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0).val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_);
  EXPECT_FLOAT_EQ(0.25,I(0).d_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_);
}

TEST(AgradFwdMatrixMdivideRightTri,fd_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_right_tri;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  row_vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf1, fd2), std::invalid_argument);
}

TEST(AgradFwdMatrixMdivideRightTri,matrix_ffd_matrix_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.0763888889,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.0625,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.5486111,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.1875,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_ffd_row_vector_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_tri;

  matrix_ffd Av(2,2);
  row_vector_ffd Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.0763888889,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.0625,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_ffd_matrix_d_lower) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.416666667,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.1875,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(-0.7986111,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.4375,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_ffd_row_vector_d_lower) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_tri;

  matrix_ffd Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.41666667,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(-0.3263889,I(0).d_.val_);
  EXPECT_FLOAT_EQ(-0.1875,I(1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_d_matrix_ffd_lower) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_tri;

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

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.41666667,I(0,0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_);
  EXPECT_FLOAT_EQ(1.0833334,I(1,0).val_.val_);
  EXPECT_FLOAT_EQ(1.75,I(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val_);
}
TEST(AgradFwdMatrixMdivideRightTri,matrix_d_row_vector_ffd_lower) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_tri;

  matrix_d Av(2,2);
  row_vector_ffd Ad(2);
  row_vector_ffd I;

  Av << 3.0, 0.0, 
    1.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Av);
  EXPECT_FLOAT_EQ(0.41666667,I(0).val_.val_);
  EXPECT_FLOAT_EQ(0.75,I(1).val_.val_);
  EXPECT_FLOAT_EQ(0.25,I(0).d_.val_);
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val_);
}

TEST(AgradFwdMatrixMdivideRightTri,ffd_exceptions_lower) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_right_tri;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  row_vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  row_vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right_tri<Eigen::Lower>(vf1, fd2), std::invalid_argument);
}
