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
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

using stan::math::fvar;
using stan::math::var;
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_matrix_fv1) {
  using stan::math::matrix_fv;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.1388889,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.10416666,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.80555558,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.60416669,I(1,1).d_.val());

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
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_matrix_fv2) {
  using stan::math::matrix_fv;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

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
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_row_vector_fv1) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  row_vector_fv Ad(2);
  row_vector_fv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.1388889,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.10416666,I(1).d_.val());

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
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_row_vector_fv2) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  row_vector_fv Ad(2);
  row_vector_fv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_matrix_d1) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.1388888,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.85416669,I(1,1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_matrix_d2) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_row_vector_d1) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  row_vector_d Ad(2);
  row_vector_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_fv_row_vector_d2) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_fv Av(2,2);
  row_vector_d Ad(2);
  row_vector_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_fv1) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(1,1).d_.val());

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
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_fv2) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_fv Ad(2,2);
  matrix_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

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
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_fv1) {
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_fv Ad(2);
  row_vector_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val());
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_fv2) {
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_fv Ad(2);
  row_vector_fv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_ = 1.0;
  Ad(1).d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_);
  vars.push_back(Ad(1).val_);
  std::vector<double> grads;

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixMdivideRightSPD,fv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right_spd;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  row_vector_fv vf1(3), vf2(4);
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

TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_ffv1) {
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.1388889,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.10416666,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.80555558,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.60416669,I(1,1).d_.val_.val());

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
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_ffv2) {
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

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
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_ffv3) {
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

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
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_ffv4) {
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

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
  EXPECT_FLOAT_EQ(-0.1712963, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.2488426, grads[2]);
  EXPECT_FLOAT_EQ(-0.090277776, grads[3]);
  EXPECT_FLOAT_EQ(0.12962963, grads[4]);
  EXPECT_FLOAT_EQ(0.097222224, grads[5]);
  EXPECT_FLOAT_EQ(0, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_ffv1) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

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
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.1388889,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.10416666,I(1).d_.val_.val());

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
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.33333334, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_ffv2) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

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

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_ffv3) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12037037, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.1736111, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
  EXPECT_FLOAT_EQ(-0.11111111, grads[4]);
  EXPECT_FLOAT_EQ(-0.083333336, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_ffv4) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.1712963, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.2488426, grads[2]);
  EXPECT_FLOAT_EQ(-0.090277776, grads[3]);
  EXPECT_FLOAT_EQ(0.12962963, grads[4]);
  EXPECT_FLOAT_EQ(0.097222224, grads[5]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_d1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-1.1388888,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.85416669,I(1,1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_d2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_d3) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_matrix_d4) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.375, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.45717594, grads[2]);
  EXPECT_FLOAT_EQ(-0.13194445, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_d1) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.22222222, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.25, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_d2) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Av(0,0).d_.val_ = 1.0;
  Av(0,1).d_.val_ = 1.0;
  Av(1,0).d_.val_ = 1.0;
  Av(1,1).d_.val_ = 1.0;
  Ad << 2.0, 3.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.47222221,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.35416666,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_d3) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.23148148, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.25694445, grads[2]);
  EXPECT_FLOAT_EQ(0.0625, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_ffv_row_vector_d4) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;

  matrix_ffv Av(2,2);
  row_vector_d Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.375, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.45717594, grads[2]);
  EXPECT_FLOAT_EQ(-0.13194445, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_ffv1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.75,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6666666,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(7.0/4.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(1,0).d_.val_.val());
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
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_ffv2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

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
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_ffv3) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

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
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_matrix_ffv4) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  matrix_ffv Ad(2,2);
  matrix_ffv I;

  Av << 3.0, 0.0, 
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

  I = mdivide_right_spd(Ad,Av);

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
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_ffv1) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);
  EXPECT_FLOAT_EQ(0.66666669,I(0).val_.val_.val());
  EXPECT_FLOAT_EQ(3.0/4.0,I(1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.33333334,I(0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.25,I(1).d_.val_.val());

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.33333334, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_ffv2) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_ffv3) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixMdivideRightSPD,matrix_d_row_vector_ffv4) {
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_d Av(2,2);
  row_vector_ffv Ad(2);
  row_vector_ffv I;

  Av << 3.0, 0.0, 
    0.0, 4.0;
  Ad << 2.0, 3.0;
  Ad(0).d_.val_ = 1.0;
  Ad(1).d_.val_ = 1.0;
  Ad(0).val_.d_ = 1.0;
  Ad(1).val_.d_ = 1.0;

  I = mdivide_right_spd(Ad,Av);

  std::vector<var> vars;
  vars.push_back(Ad(0).val_.val_);
  vars.push_back(Ad(1).val_.val_);
  std::vector<double> grads;

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixMdivideRightSPD,ffv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right_spd;

  matrix_ffv fv1(3,3), fv2(4,4);
  row_vector_ffv rvf1(3), rvf2(4);
  row_vector_ffv vf1(3), vf2(4);
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
