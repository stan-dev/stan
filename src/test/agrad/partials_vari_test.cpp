#include <gtest/gtest.h>

#include <stan/agrad/partials_vari.hpp>
#include <stan/math/special_functions.hpp>


TEST(AgradPartialsVari,simple_var_vvv) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  var a = 1.0;  double da = 5.0;
  var b = 2.0;  double db = 7.0;
  var c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(5.0,a.adj());
  EXPECT_FLOAT_EQ(7.0,b.adj());
  EXPECT_FLOAT_EQ(11.0,c.adj());
}
TEST(AgradPartialsVari,simple_var_vvd) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  var a = 1.0;  double da = 5.0;
  var b = 2.0;  double db = 7.0;
  double c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(5.0,a.adj());
  EXPECT_FLOAT_EQ(7.0,b.adj());
}
TEST(AgradPartialsVari,simple_var_vdv) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  var a = 1.0;  double da = 5.0;
  double b = 2.0;  double db = 7.0;
  var c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(5.0,a.adj());
  EXPECT_FLOAT_EQ(11.0,c.adj());
}
TEST(AgradPartialsVari,simple_var_vdd) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  var a = 1.0;  double da = 5.0;
  double b = 2.0;  double db = 7.0;
  double c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(5.0,a.adj());
}
TEST(AgradPartialsVari,simple_var_dvv) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  double a = 1.0;  double da = 5.0;
  var b = 2.0;  double db = 7.0;
  var c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(7.0,b.adj());
  EXPECT_FLOAT_EQ(11.0,c.adj());
}
TEST(AgradPartialsVari,simple_var_dvd) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  double a = 1.0;  double da = 5.0;
  var b = 2.0;  double db = 7.0;
  double c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(7.0,b.adj());
}
TEST(AgradPartialsVari,simple_var_ddv) {
  using stan::agrad::var;
  using stan::agrad::simple_var;
  double y = -1.0;
  double a = 1.0;  double da = 5.0;
  double b = 2.0;  double db = 7.0;
  var c = 3.0;  double dc = 11.0;
  var v = simple_var(y,a,da,b,db,c,dc);
  stan::agrad::grad(v.vi_);
  EXPECT_FLOAT_EQ(-1.0,v.val());
  EXPECT_FLOAT_EQ(11.0,c.adj());
}


TEST(AgradPartialsVari,partials1_vari) {
  using stan::agrad::var;
  using stan::agrad::partials1_vari;
  var x = 2.0;
  var z = -3.0 * x;  // dz/dx = -3
  var y(new partials1_vari(-1.0,z.vi_,5.0)); // dy/dz = 5
  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-15.0, x.adj());  // dy/dx = -15
}
TEST(AgradPartialsVari,partials2_vari) {
  using stan::agrad::var;
  using stan::agrad::vari;
  using stan::agrad::partials2_vari;
  var x1 = 2.0;
  var x2 = 3.0;
  var z1 = -5.0 * x1; // dz1/dx1 = -5
  var z2 = -7.0 * x2; // dz2/dx2 = -7
  var y(new partials2_vari<vari*, vari*>(-1.0,
                                         z1.vi_,11.0,   // dy/dz1 = 11.0
                                         z2.vi_,13.0)); // dy/dz2 = 13.0
  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-55.0, x1.adj());  // dy/dx1 = -55
  EXPECT_FLOAT_EQ(-91.0, x2.adj());  // dy/dx2 = -91
}
TEST(AgradPartialsVari,partials3_vari) {
  using stan::agrad::var;
  using stan::agrad::vari;
  using stan::agrad::partials3_vari;
  var x1 = 2.0;
  var x2 = 3.0;
  var x3 = 5.0;
  var z1 = -7.0 * x1;  // dz1/dx1 = -5
  var z2 = -9.0 * x2;  // dz2/dx2 = -7
  var z3 = -11.0 * x3; // dz3/dx3 = -11
  var y(new partials3_vari<vari*, vari*, vari*>(-1.0,
                                                z1.vi_,17.0,   // dy/dz1 = 17.0
                                                z2.vi_,19.0,   // dy/dz2 = 19.0
                                                z3.vi_,23.0)); // dy/dz3 = 23.0
  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
}
