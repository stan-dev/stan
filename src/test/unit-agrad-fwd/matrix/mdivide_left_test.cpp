#include <stan/agrad/fwd/matrix/mdivide_left.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_left.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrixMdivideLeft,fd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::mdivide_left;

  matrix_fd Av(2,2);
  matrix_d Ad(2,2);
  matrix_fd I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_,1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(8.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(8.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(-6.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(-6.0,I(1,1).d_,1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_,1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_,1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_,1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_,1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_,1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_,1.0e-12);
}
TEST(AgradFwdMatrixMdivideLeft,fd_matrix_vector) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_fd fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_fd vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_fd output;
  output = mdivide_left(fv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_,1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(1.0,output(1,0).d_,1.0E-12);

  output = mdivide_left(fv, vecd);
  EXPECT_NEAR(-4.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_,1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(-1.0,output(1,0).d_,1.0E-12);

  output = mdivide_left(dv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_,1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_,1.0E-12);
  EXPECT_NEAR(-2.0,output(0,0).d_,1.0E-12);
  EXPECT_NEAR(2.0,output(1,0).d_,1.0E-12);
}
TEST(AgradFwdMatrixMdivideLeft,fd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::agrad::mdivide_left;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::domain_error);
}
TEST(AgradFwdMatrixMdivideLeft,fv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::mdivide_left;

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

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(8.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(8.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_.val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val_,Av(0,1).val_,Av(1,0).val_,Av(1,1).val_);
  VEC h;
  I(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-7.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(3.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,fv_matrix_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::mdivide_left;

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

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_.val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val_,Av(0,1).val_,Av(1,0).val_,Av(1,1).val_);
  VEC h;
  I(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,fv_matrix_vector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_fv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_fv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_fv output;
  output = mdivide_left(fv, vecd);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(1,0).d_.val(),1.0E-12);

  output = mdivide_left(dv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(2.0,output(1,0).d_.val(),1.0E-12);

  output = mdivide_left(fv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(1,0).d_.val(),1.0E-12);

  AVEC q = createAVEC(fv(0,0).val_,fv(0,1).val_,fv(1,0).val_,fv(1,1).val_);
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-8.0,h[0]);
  EXPECT_FLOAT_EQ(9.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(-4.5,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,fv_matrix_vector_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_fv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_fv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_fv output;
  output = mdivide_left(fv, vecf);

  AVEC q = createAVEC(fv(0,0).val_,fv(0,1).val_,fv(1,0).val_,fv(1,1).val_);
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-6.0,h[0]);
  EXPECT_FLOAT_EQ(6.5,h[1]);
  EXPECT_FLOAT_EQ(5.0,h[2]);
  EXPECT_FLOAT_EQ(-5.5,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,fv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::agrad::mdivide_left;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::domain_error);
}
TEST(AgradFwdMatrixMdivideLeft,ffd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::mdivide_left;

  matrix_ffd Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffd I;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 2.0;
  b.val_.val_ = 3.0;
  c.val_.val_ = 5.0;
  d.val_.val_ = 7.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;

  Av << a,b,c,d;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(8.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(8.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_.val(),1.0e-12);
}
TEST(AgradFwdMatrixMdivideLeft,ffd_matrix_vector) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = 5.0;
  f.val_.val_ = 6.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;

  matrix_ffd fv(2,2);
  fv << a,b,c,d;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_ffd vecf(2);
  vecf << e,f;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_ffd output;
  output = mdivide_left(fv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(1,0).d_.val(),1.0E-12);

  output = mdivide_left(fv, vecd);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(1,0).d_.val(),1.0E-12);

  output = mdivide_left(dv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(2.0,output(1,0).d_.val(),1.0E-12);
}
TEST(AgradFwdMatrixMdivideLeft,ffd_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::mdivide_left;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::domain_error);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::mdivide_left;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_.val().val(),1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(8.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(8.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-6.0,I(1,1).d_.val().val(),1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_.val().val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val_.val(),Av(0,1).val_.val(),Av(1,0).val_.val(),Av(1,1).val_.val());
  VEC h;
  I(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-7.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(3.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::mdivide_left;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val_.val(),Av(0,1).val_.val(),Av(1,0).val_.val(),Av(1,1).val_.val());
  VEC h;
  I(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::mdivide_left;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(-8.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-8.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(6.0,I(1,1).d_.val().val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val_.val(),Av(0,1).val_.val(),Av(1,0).val_.val(),Av(1,1).val_.val());
  VEC h;
  I(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::mdivide_left;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 2.0;
  Av(0,1).d_ = 2.0;
  Av(1,0).d_ = 2.0;
  Av(1,1).d_ = 2.0;
  Av(0,0).val_.d_ = 2.0;
  Av(0,1).val_.d_ = 2.0;
  Av(1,0).val_.d_ = 2.0;
  Av(1,1).val_.d_ = 2.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val_.val(),Av(0,1).val_.val(),Av(1,0).val_.val(),Av(1,1).val_.val());
  VEC h;
  I(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_vector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_ffv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_ffv output;
  output = mdivide_left(fv, vecd);
  EXPECT_NEAR(-4.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(1,0).d_.val().val(),1.0E-12);

  output = mdivide_left(dv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(-2.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,output(1,0).d_.val().val(),1.0E-12);

  output = mdivide_left(fv, vecf);
  EXPECT_NEAR(-4.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(4.5,output(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,output(1,0).d_.val().val(),1.0E-12);

  AVEC q = createAVEC(fv(0,0).val_.val(),fv(0,1).val_.val(),fv(1,0).val_.val(),fv(1,1).val_.val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-8.0,h[0]);
  EXPECT_FLOAT_EQ(9.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(-4.5,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_vector_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_ffv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_ffv output;
  output = mdivide_left(fv, vecf);

  AVEC q = createAVEC(fv(0,0).val_.val(),fv(0,1).val_.val(),fv(1,0).val_.val(),fv(1,1).val_.val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_vector_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_ffv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_ffv output;
  output = mdivide_left(fv, vecf);

  AVEC q = createAVEC(fv(0,0).val_.val(),fv(0,1).val_.val(),fv(1,0).val_.val(),fv(1,1).val_.val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-6.0,h[0]);
  EXPECT_FLOAT_EQ(6.5,h[1]);
  EXPECT_FLOAT_EQ(5.0,h[2]);
  EXPECT_FLOAT_EQ(-5.5,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_matrix_vector_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::mdivide_left;

  matrix_ffv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 1.0;
  fv(0,1).d_ = 1.0;
  fv(1,0).d_ = 1.0;
  fv(1,1).d_ = 1.0;
  fv(0,0).val_.d_ = 1.0;
  fv(0,1).val_.d_ = 1.0;
  fv(1,0).val_.d_ = 1.0;
  fv(1,1).val_.d_ = 1.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 1.0;
  vecf(1).d_ = 1.0;
  vecf(0).val_.d_ = 1.0;
  vecf(1).val_.d_ = 1.0;

  vector_d vecd(2);
  vecd << 5,6;

  matrix_ffv output;
  output = mdivide_left(fv, vecf);

  AVEC q = createAVEC(fv(0,0).val_.val(),fv(0,1).val_.val(),fv(1,0).val_.val(),fv(1,1).val_.val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0.5,h[1]);
  EXPECT_FLOAT_EQ(0.5,h[2]);
  EXPECT_FLOAT_EQ(-0.5,h[3]);
}
TEST(AgradFwdMatrixMdivideLeft,ffv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::mdivide_left;

  matrix_ffv fv1(3,3), fv2(4,4);
  row_vector_ffv rvf1(3), rvf2(4);
  vector_ffv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::domain_error);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::domain_error);
}
