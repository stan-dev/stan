#include <stan/diff/operands_and_partials.hpp>
#include <gtest/gtest.h>
#include <stan/diff.hpp>



TEST(DiffPartialsVari, OperandsAndPartials) {
  using stan::diff::OperandsAndPartials;
  using stan::diff::var;

  OperandsAndPartials<double> o1;
  EXPECT_EQ(0U, o1.nvaris);
  
  OperandsAndPartials<double,double,double,double> o2;
  EXPECT_EQ(0U, o2.nvaris);
  
  std::vector<double> d_vec(4);
  OperandsAndPartials<std::vector<double> > o3(d_vec);
  EXPECT_EQ(0U, o3.nvaris);

  std::vector<var> v_vec;
  v_vec.push_back(var(0.0));
  v_vec.push_back(var(1.0));
  v_vec.push_back(var(2.0));
  v_vec.push_back(var(3.0));
  
  std::vector<double> grad;

  OperandsAndPartials<std::vector<var> > o4(v_vec);
  o4.d_x1[0] = 10.0;
  o4.d_x1[1] = 20.0;
  o4.d_x1[2] = 30.0;
  o4.d_x1[3] = 40.0;
  
  var v = o4.to_var(10.0);
  v.grad(v_vec, grad);
  EXPECT_EQ(4U, o4.nvaris);
  EXPECT_FLOAT_EQ(10.0, v.val());
  EXPECT_FLOAT_EQ(10.0, grad[0]);
  EXPECT_FLOAT_EQ(20.0, grad[1]);
  EXPECT_FLOAT_EQ(30.0, grad[2]);
  EXPECT_FLOAT_EQ(40.0, grad[3]);
}
TEST(DiffPartialsVari,OperandsAndPartials1) {
  using stan::diff::vari;
  using stan::diff::OperandsAndPartials;
  using stan::diff::var;

  var x = 2.0;
  var z = -3.0 * x;  // dz/dx = -3
  OperandsAndPartials<var> o(z);
  o.d_x1[0] += 5.0;  // dy/dz = 5

  var y = o.to_var(-1.0);

  stan::diff::grad(y.vi_);
  EXPECT_FLOAT_EQ(-15.0, x.adj());  // dy/dx = -15
}
TEST(DiffPartialsVari,OperandsAndPartials2) {
  using stan::diff::OperandsAndPartials;
  using stan::diff::vari;
  using stan::diff::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var z1 = -5.0 * x1; // dz1/dx1 = -5
  var z2 = -7.0 * x2; // dz2/dx2 = -7
  OperandsAndPartials<var,var> o(z1,z2);
  o.d_x1[0] += 11.0;  // dy/dz1 = 11.0
  o.d_x2[0] += 13.0;  // dy/dz2 = 13.0
  var y = o.to_var(-1.0);

  stan::diff::grad(y.vi_);
  EXPECT_FLOAT_EQ(-55.0, x1.adj());  // dy/dx1 = -55
  EXPECT_FLOAT_EQ(-91.0, x2.adj());  // dy/dx2 = -91
}
TEST(DiffPartialsVari, OperandsAndPartials3) {
  using stan::diff::OperandsAndPartials;
  using stan::diff::vari;
  using stan::diff::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var x3 = 5.0;
  var z1 = -7.0 * x1;  // dz1/dx1 = -5
  var z2 = -9.0 * x2;  // dz2/dx2 = -7
  var z3 = -11.0 * x3; // dz3/dx3 = -11

  OperandsAndPartials<var,var,var> o(z1, z2, z3);
  o.d_x1[0] += 17.0;  // dy/dz1 = 17.0
  o.d_x2[0] += 19.0;  // dy/dz2 = 19.0
  o.d_x3[0] += 23.0;  // dy/dz3 = 23.0
  var y = o.to_var(-1.0);

  stan::diff::grad(y.vi_);
  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
}
