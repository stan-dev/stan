#include <stan/agrad/fwd/matrix/mdivide_right.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_left.hpp>
#include <stan/math/matrix/multiply.hpp>

TEST(AgradFwdMatrix,mdivide_right_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::mdivide_right;

  matrix_fv Av(2,2);
  matrix_d Ad(2,2);
  matrix_fv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_,1.0e-12);

  I = mdivide_right(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(-4.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(2.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(-4.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(2.0,I(1,1).d_,1.0e-12);

  I = mdivide_right(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(4.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(-2.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(4.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(-2.0,I(1,1).d_,1.0e-12);
}
TEST(AgradFwdMatrix,mdivide_right_matrix_rowvector) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::mdivide_right;

  matrix_fv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_fv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  row_vector_d vecd(2);
  vecd << 5,6;

  matrix_fv output;
  output = mdivide_right(vecf,fv);
  EXPECT_NEAR(-1.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(0.0,output(0,1).d_,1.0E-12);

  output = mdivide_right(vecd,fv);
  EXPECT_NEAR(-1.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_,1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(-1.0,output(0,1).d_,1.0E-12);

  output = mdivide_right(vecf,dv);
  EXPECT_NEAR(-1.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_,1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(1.0,output(0,1).d_,1.0E-12);
}
TEST(AgradFwdMatrix,mdivide_right_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::agrad::mdivide_right;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(rvf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right(rvd2, fv1), std::domain_error);  
  EXPECT_THROW(mdivide_right(vd1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right(vd2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right(vf1, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right(vf2, fv1), std::domain_error);
  EXPECT_THROW(mdivide_right(rvf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(rvd1, fv2), std::domain_error);  
  EXPECT_THROW(mdivide_right(vd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(vd2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(vf1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(vf2, fv2), std::domain_error);
  EXPECT_THROW(mdivide_right(rvf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right(vf1, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right(vf2, fd1), std::domain_error);
  EXPECT_THROW(mdivide_right(rvf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right(vf1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_right(vf2, fd2), std::domain_error);
}
