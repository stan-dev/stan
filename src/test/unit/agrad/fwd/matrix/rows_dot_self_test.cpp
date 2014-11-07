#include <stan/agrad/fwd/matrix/rows_dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixRowsDotSelf,fd) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).val_,1E-12);
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).d_,1E-12);

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> x;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_,1E-12);

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);
  EXPECT_NEAR(13.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(41.0,x(1,0).val_,1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_,1E-12);
  EXPECT_NEAR(18.0,x(1,0).d_,1E-12);
}
TEST(AgradFwdMatrixRowsDotSelf,fv_1stDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).d_.val(),1E-12);

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> x;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val(),1E-12);

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);
  EXPECT_NEAR(13.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(41.0,x(1,0).val_.val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(18.0,x(1,0).d_.val(),1E-12);

  AVEC q = createAVEC(m3(0,0).val(),m3(0,1).val(),m3(1,0).val(),m3(1,1).val());
  VEC h;
  x(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(6,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixRowsDotSelf,fv_2ndDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> x;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);

  AVEC q = createAVEC(m3(0,0).val(),m3(0,1).val(),m3(1,0).val(),m3(1,1).val());
  VEC h;
  x(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixRowsDotSelf,ffd) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).d_.val(),1E-12);

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> x;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val(),1E-12);

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);
  EXPECT_NEAR(13.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(41.0,x(1,0).val_.val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(18.0,x(1,0).d_.val(),1E-12);
}
TEST(AgradFwdMatrixRowsDotSelf,ffv_1stDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0).d_.val().val(),1E-12);

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> x;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val().val(),1E-12);

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);
  EXPECT_NEAR(13.0,x(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(41.0,x(1,0).val_.val().val(),1E-12);
  EXPECT_NEAR(10.0,x(0,0).d_.val().val(),1E-12);
  EXPECT_NEAR(18.0,x(1,0).d_.val().val(),1E-12);

  AVEC q = createAVEC(m3(0,0).val().val(),m3(0,1).val().val(),m3(1,0).val().val(),m3(1,1).val().val());
  VEC h;
  x(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(6,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixRowsDotSelf,ffv_2ndDeriv_1) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> x;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);

  AVEC q = createAVEC(m3(0,0).val().val(),m3(0,1).val().val(),m3(1,0).val().val(),m3(1,1).val().val());
  VEC h;
  x(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradFwdMatrixRowsDotSelf,ffv_2ndDeriv_2) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> x;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = rows_dot_self(m3);

  AVEC q = createAVEC(m3(0,0).val().val(),m3(0,1).val().val(),m3(1,0).val().val(),m3(1,1).val().val());
  VEC h;
  x(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixRowsDotSelf,ffv_3rdDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> x;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  m3(0,0).val_.d_ = 1.0;
  m3(0,1).val_.d_ = 1.0;
  m3(1,0).val_.d_ = 1.0;
  m3(1,1).val_.d_ = 1.0;
  x = rows_dot_self(m3);

  AVEC q = createAVEC(m3(0,0).val().val(),m3(0,1).val().val(),m3(1,0).val().val(),m3(1,1).val().val());
  VEC h;
  x(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

