#include <stan/math/fwd/mat/fun/mdivide_right.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/mdivide_left.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
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
TEST(AgradMixMatrixMdivideRight,fv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::mdivide_right;

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
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_right(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(-4.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(2.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(-4.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(2.0,I(1,1).d_.val(),1.0e-12);

  I = mdivide_right(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val(),1.0e-12);
  EXPECT_NEAR(4.0,I(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-2.0,I(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR(4.0,I(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-2.0,I(1,1).d_.val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val(),Av(0,1).val(),Av(1,0).val(),Av(1,1).val());
  VEC h;
  I(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(7.0,h[0]);
  EXPECT_FLOAT_EQ(-5.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixMdivideRight,fv_matrix_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::mdivide_right;

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

  I = mdivide_right(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val(),Av(0,1).val(),Av(1,0).val(),Av(1,1).val());
  VEC h;
  I(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(44.0,h[0]);
  EXPECT_FLOAT_EQ(-32.0,h[1]);
  EXPECT_FLOAT_EQ(-14.0,h[2]);
  EXPECT_FLOAT_EQ(10.0,h[3]);
}
TEST(AgradMixMatrixMdivideRight,fv_matrix_rowvector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

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
  EXPECT_NEAR(-1.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(0.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(0.0,output(0,1).d_.val(),1.0E-12);

  output = mdivide_right(vecd,fv);
  EXPECT_NEAR(-1.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,1).d_.val(),1.0E-12);

  output = mdivide_right(vecf,dv);
  EXPECT_NEAR(-1.0,output(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,1).d_.val(),1.0E-12);

  AVEC q = createAVEC(vecf(0).val(),vecf(1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2.0,h[0]);
  EXPECT_FLOAT_EQ(1.5,h[1]);
}
TEST(AgradMixMatrixMdivideRight,fv_matrix_rowvector_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_fv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  matrix_fv output;
  output = mdivide_right(vecf,dv);

  AVEC q = createAVEC(vecf(0).val(),vecf(1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradMixMatrixMdivideRight,fv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;
  using stan::math::mdivide_right;

  matrix_fv fv1(3,3), fv2(4,4);
  row_vector_fv rvf1(3), rvf2(4);
  vector_fv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvd2, fv1), std::invalid_argument);  
  EXPECT_THROW(mdivide_right(vd1, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvd1, fv2), std::invalid_argument);  
  EXPECT_THROW(mdivide_right(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vd2, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fd2), std::invalid_argument);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right;

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

  I = mdivide_right(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(0.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,1).d_.val().val(),1.0e-12);

  I = mdivide_right(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(-4.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-4.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,I(1,1).d_.val().val(),1.0e-12);

  I = mdivide_right(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val_.val().val(),1.0e-12);
  EXPECT_NEAR(4.0,I(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-2.0,I(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR(4.0,I(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-2.0,I(1,1).d_.val().val(),1.0e-12);

  AVEC q = createAVEC(Av(0,0).val().val(),Av(0,1).val().val(),Av(1,0).val().val(),Av(1,1).val().val());
  VEC h;
  I(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(7.0,h[0]);
  EXPECT_FLOAT_EQ(-5.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right;

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

  I = mdivide_right(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val().val(),Av(0,1).val().val(),Av(1,0).val().val(),Av(1,1).val().val());
  VEC h;
  I(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right;

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

  I = mdivide_right(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val().val(),Av(0,1).val().val(),Av(1,0).val().val(),Av(1,1).val().val());
  VEC h;
  I(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(44.0,h[0]);
  EXPECT_FLOAT_EQ(-32.0,h[1]);
  EXPECT_FLOAT_EQ(-14.0,h[2]);
  EXPECT_FLOAT_EQ(10.0,h[3]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::mdivide_right;

  matrix_ffv Av(2,2);
  matrix_d Ad(2,2);
  matrix_ffv I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Av(0,0).d_ = 1.0;
  Av(0,1).d_ = 1.0;
  Av(1,0).d_ = 1.0;
  Av(1,1).d_ = 1.0;
  Av(0,0).val_.d_ = 1.0;
  Av(0,1).val_.d_ = 1.0;
  Av(1,0).val_.d_ = 1.0;
  Av(1,1).val_.d_ = 1.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right(Ad,Av);

  AVEC q = createAVEC(Av(0,0).val().val(),Av(0,1).val().val(),Av(1,0).val().val(),Av(1,1).val().val());
  VEC h;
  I(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(76,h[0]);
  EXPECT_FLOAT_EQ(-56,h[1]);
  EXPECT_FLOAT_EQ(-30,h[2]);
  EXPECT_FLOAT_EQ(22,h[3]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_rowvector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

  matrix_ffv fv(2,2);
  fv << 1, 2, 3, 4;
  fv(0,0).d_ = 2.0;
  fv(0,1).d_ = 2.0;
  fv(1,0).d_ = 2.0;
  fv(1,1).d_ = 2.0;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  row_vector_d vecd(2);
  vecd << 5,6;

  matrix_ffv output;
  output = mdivide_right(vecf,fv);
  EXPECT_NEAR(-1.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(0.0,output(0,1).d_.val().val(),1.0E-12);

  output = mdivide_right(vecd,fv);
  EXPECT_NEAR(-1.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,1).d_.val().val(),1.0E-12);

  output = mdivide_right(vecf,dv);
  EXPECT_NEAR(-1.0,output(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(2.0,output(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(-1.0,output(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(1.0,output(0,1).d_.val().val(),1.0E-12);

  AVEC q = createAVEC(vecf(0).val().val(),vecf(1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2.0,h[0]);
  EXPECT_FLOAT_EQ(1.5,h[1]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_rowvector_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  matrix_ffv output;
  output = mdivide_right(vecf,dv);

  AVEC q = createAVEC(vecf(0).val().val(),vecf(1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_rowvector_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 2.0;
  vecf(1).d_ = 2.0;

  matrix_ffv output;
  output = mdivide_right(vecf,dv);

  AVEC q = createAVEC(vecf(0).val().val(),vecf(1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradMixMatrixMdivideRight,ffv_matrix_rowvector_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right;

  matrix_d dv(2,2);
  dv << 1, 2, 3, 4;

  row_vector_ffv vecf(2);
  vecf << 5, 6;
  vecf(0).d_ = 1.0;
  vecf(1).d_ = 1.0;
  vecf(0).val_.d_ = 1.0;
  vecf(1).val_.d_ = 1.0;

  matrix_ffv output;
  output = mdivide_right(vecf,dv);

  AVEC q = createAVEC(vecf(0).val().val(),vecf(1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradMixMatrixMdivideRight,ffv_exceptions) {
  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::mdivide_right;

  matrix_ffv fv1(3,3), fv2(4,4);
  row_vector_ffv rvf1(3), rvf2(4);
  vector_ffv vf1(3), vf2(4);
  matrix_d fd1(3,3), fd2(4,4);
  row_vector_d rvd1(3), rvd2(4);
  vector_d vd1(3), vd2(4);

  EXPECT_THROW(mdivide_right(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fv1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(fd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvd2, fv1), std::invalid_argument);  
  EXPECT_THROW(mdivide_right(vd1, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vd2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvd1, fv2), std::invalid_argument);  
  EXPECT_THROW(mdivide_right(vd1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vd2, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fd1), std::invalid_argument);
  EXPECT_THROW(mdivide_right(rvf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf1, fd2), std::invalid_argument);
  EXPECT_THROW(mdivide_right(vf2, fd2), std::invalid_argument);
}
