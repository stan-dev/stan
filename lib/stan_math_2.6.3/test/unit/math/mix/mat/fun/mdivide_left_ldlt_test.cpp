#include <stan/math/prim/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/rev/mat/fun/mdivide_left_ldlt.hpp>
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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_matrix_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  stan::math::matrix_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_.val());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.12,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.8,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.28,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).d_.val());
  
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
  EXPECT_FLOAT_EQ(0.28, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.24, grads[2]);
  EXPECT_FLOAT_EQ(0.48, grads[3]);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.6, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_matrix_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  stan::math::matrix_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  
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
  EXPECT_FLOAT_EQ(-1.696, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.608, grads[2]);
  EXPECT_FLOAT_EQ(-0.296, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0.16, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.48,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.12,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(0.2,I(1,1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.28, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.24, grads[2]);
  EXPECT_FLOAT_EQ(0.48, grads[3]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_);
  vars.push_back(Ad(0,1).val_);
  vars.push_back(Ad(1,0).val_);
  vars.push_back(Ad(1,1).val_);

  I(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.544, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.088, grads[2]);
  EXPECT_FLOAT_EQ(-0.056, grads[3]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_fv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  stan::math::matrix_fv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_);
  vars.push_back(Av(0,1).val_);
  vars.push_back(Av(1,0).val_);
  vars.push_back(Av(1,1).val_);

  I(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.6, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_fv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_fv Av(2,2);
  stan::math::matrix_fv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_vector_fv_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_fv Av(2);
  stan::math::vector_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_vector_fv_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_fv Av(2);
  stan::math::vector_fv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_vector_d_1) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_fv_vector_d_2) {
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_Ad;
  stan::math::matrix_fv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_ = 1.0;
  Ad(0,1).d_ = 1.0;
  Ad(1,0).d_ = 1.0;
  Ad(1,1).d_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_fv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_fv Av(2);
  stan::math::vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_fv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_fv Av(2);
  stan::math::vector_fv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_ = 2.0;
  Av(1).d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_);
  vars.push_back(Av(1).val_);

  I(0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,fv_exceptions) {
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

  EXPECT_THROW(mdivide_left_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, vf1), std::invalid_argument);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Ad);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(1,1).d_.val_.val());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.12,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.28,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.2,I(1,1).d_.val_.val());
  
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
  EXPECT_FLOAT_EQ(0.28, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.24, grads[2]);
  EXPECT_FLOAT_EQ(0.48, grads[3]);
  EXPECT_FLOAT_EQ(1.4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.6, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  
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
  EXPECT_FLOAT_EQ(-1.696, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.608, grads[2]);
  EXPECT_FLOAT_EQ(-0.296, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0.16, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  
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
  EXPECT_FLOAT_EQ(-1.696, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(1.608, grads[2]);
  EXPECT_FLOAT_EQ(-0.296, grads[3]);
  EXPECT_FLOAT_EQ(-0.64, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(0.16, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  
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
  EXPECT_FLOAT_EQ(3.4688, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-2.6464, grads[2]);
  EXPECT_FLOAT_EQ(0.4448, grads[3]);
  EXPECT_FLOAT_EQ(0.768, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
  EXPECT_FLOAT_EQ(-0.192, grads[6]);
  EXPECT_FLOAT_EQ(0, grads[7]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.48,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.8,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0.12,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0.2,I(1,1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.28, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-1.24, grads[2]);
  EXPECT_FLOAT_EQ(0.48, grads[3]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.544, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.088, grads[2]);
  EXPECT_FLOAT_EQ(-0.056, grads[3]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.544, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.088, grads[2]);
  EXPECT_FLOAT_EQ(-0.056, grads[3]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_matrix_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::matrix_d Av(2,2);
  stan::math::matrix_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Ad(0,0).val_.val_);
  vars.push_back(Ad(0,1).val_.val_);
  vars.push_back(Ad(1,0).val_.val_);
  vars.push_back(Ad(1,1).val_.val_);

  I(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-1.2672, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.2016, grads[2]);
  EXPECT_FLOAT_EQ(0.0288, grads[3]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_ffv_1) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_FLOAT_EQ(-0.2,I(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.0,I(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.8,I(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0,I(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1.6,I(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.4,I(1,1).d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0,0).val_.val_);
  vars.push_back(Av(0,1).val_.val_);
  vars.push_back(Av(1,0).val_.val_);
  vars.push_back(Av(1,1).val_.val_);

  I(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(1.4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.6, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_ffv_2) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_ffv_3) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_matrix_ffv_4) {
  stan::math::LDLT_factor<double ,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_ffv Av(2,2);
  stan::math::matrix_ffv I;

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
  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_ffv_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_ffv_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_ffv_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_ffv_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_d_1) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_d_2) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Ad(0,0).d_.val_ = 1.0;
  Ad(0,1).d_.val_ = 1.0;
  Ad(1,0).d_.val_ = 1.0;
  Ad(1,1).d_.val_ = 1.0;
  Av << 2.0, 3.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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

TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_d_3) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_ffv_vector_d_4) {
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_Ad;
  stan::math::matrix_ffv Ad(2,2);
  stan::math::vector_d Av(2);
  stan::math::vector_ffv I;

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

  I = mdivide_left_ldlt(ldlt_Ad,Av);

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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_ffv_1) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);
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
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_ffv_2) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_ffv_3) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixMdivideLeftLDLT,matrix_d_vector_ffv_4) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::vector_ffv Av(2);
  stan::math::vector_ffv I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;
  Av << 2.0, 3.0;
  Av(0).d_.val_ = 2.0;
  Av(1).d_.val_ = 2.0;
  Av(0).val_.d_ = 2.0;
  Av(1).val_.d_ = 2.0;

  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Ad,Av);

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(Av(0).val_.val_);
  vars.push_back(Av(1).val_.val_);

  I(0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixMdivideLeftLDLT,ffv_exceptions) {
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

  EXPECT_THROW(mdivide_left_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, fd2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, fv2_), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_ldlt(fd2, vf1), std::invalid_argument);
}
