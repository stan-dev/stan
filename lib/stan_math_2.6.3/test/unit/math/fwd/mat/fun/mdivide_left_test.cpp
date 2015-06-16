#include <stan/math/fwd/mat/fun/mdivide_left.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/mdivide_left.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixMdivideLeft,fd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::mdivide_left;

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
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::vector_d;
  using stan::math::mdivide_left;

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
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::mdivide_left;

  matrix_fd fv1(3,3), fv2(4,4);
  row_vector_fd rvf1(3), rvf2(4);
  vector_fd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::invalid_argument);
}
TEST(AgradFwdMatrixMdivideLeft,ffd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left;

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
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::vector_d;
  using stan::math::mdivide_left;

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
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::mdivide_left;

  matrix_ffd fv1(3,3), fv2(4,4);
  row_vector_ffd rvf1(3), rvf2(4);
  vector_ffd vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_left(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv1, vd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvd2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, rvd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, vf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fv2, vd1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd1, vf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, rvf2), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, rvf1), std::invalid_argument);
  EXPECT_THROW(mdivide_left(fd2, vf1), std::invalid_argument);
}
