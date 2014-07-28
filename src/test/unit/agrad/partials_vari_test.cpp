#include <stan/agrad/partials_vari.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>



TEST(AgradPartialsVari, OperandsAndPartials) {
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::var;

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
  
  var v = o4.to_var(10.0,v_vec);
  v.grad(v_vec, grad);
  EXPECT_EQ(4U, o4.nvaris);
  EXPECT_FLOAT_EQ(10.0, v.val());
  EXPECT_FLOAT_EQ(10.0, grad[0]);
  EXPECT_FLOAT_EQ(20.0, grad[1]);
  EXPECT_FLOAT_EQ(30.0, grad[2]);
  EXPECT_FLOAT_EQ(40.0, grad[3]);
}
TEST(AgradPartialsVari,OperandsAndPartials1) {
  using stan::agrad::vari;
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::var;

  var x = 2.0;
  var z = -3.0 * x;  // dz/dx = -3
  OperandsAndPartials<var> o(z);
  o.d_x1[0] += 5.0;  // dy/dz = 5

  var y = o.to_var(-1.0);

  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-15.0, x.adj());  // dy/dx = -15
}
TEST(AgradPartialsVari,OperandsAndPartials2) {
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::vari;
  using stan::agrad::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var z1 = -5.0 * x1; // dz1/dx1 = -5
  var z2 = -7.0 * x2; // dz2/dx2 = -7
  OperandsAndPartials<var,var> o(z1,z2);
  o.d_x1[0] += 11.0;  // dy/dz1 = 11.0
  o.d_x2[0] += 13.0;  // dy/dz2 = 13.0
  var y = o.to_var(-1.0);

  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-55.0, x1.adj());  // dy/dx1 = -55
  EXPECT_FLOAT_EQ(-91.0, x2.adj());  // dy/dx2 = -91
}
TEST(AgradPartialsVari, OperandsAndPartials3) {
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::vari;
  using stan::agrad::var;

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

  stan::agrad::grad(y.vi_);
  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
}
TEST(AgradPartialsVari, OperandsAndPartials_check_throw) {
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::var;
  using std::vector;
  
  double d;
  vector<double> D;
  var v;
  vector<var> V;
  
  OperandsAndPartials<> o1(d,d,d,d,d,d);
  EXPECT_THROW(o1.d_x1[0], std::out_of_range);
  EXPECT_THROW(o1.d_x2[0], std::out_of_range);
  EXPECT_THROW(o1.d_x3[0], std::out_of_range);
  EXPECT_THROW(o1.d_x4[0], std::out_of_range);
  EXPECT_THROW(o1.d_x5[0], std::out_of_range);
  EXPECT_THROW(o1.d_x6[0], std::out_of_range);

  OperandsAndPartials<var,var,var,var,var,var> o2(v,v,v,v,v,v);
  EXPECT_NO_THROW(o2.d_x1[0]);
  EXPECT_NO_THROW(o2.d_x2[0]);
  EXPECT_NO_THROW(o2.d_x3[0]);
  EXPECT_NO_THROW(o2.d_x4[0]);
  EXPECT_NO_THROW(o2.d_x5[0]);
  EXPECT_NO_THROW(o2.d_x6[0]);

  OperandsAndPartials<vector<double>,vector<double>,vector<double>,
                      vector<double>,vector<double>,vector<double> > o3(D,D,D,D,D,D);
  EXPECT_THROW(o3.d_x1[0], std::out_of_range);
  EXPECT_THROW(o3.d_x2[0], std::out_of_range);
  EXPECT_THROW(o3.d_x3[0], std::out_of_range);
  EXPECT_THROW(o3.d_x4[0], std::out_of_range);
  EXPECT_THROW(o3.d_x5[0], std::out_of_range);
  EXPECT_THROW(o3.d_x6[0], std::out_of_range);

  OperandsAndPartials<vector<var>,vector<var>,vector<var>,
                      vector<var>,vector<var>,vector<var> > o4(V,V,V,V,V,V);
  EXPECT_NO_THROW(o4.d_x1[0]);
  EXPECT_NO_THROW(o4.d_x2[0]);
  EXPECT_NO_THROW(o4.d_x3[0]);
  EXPECT_NO_THROW(o4.d_x4[0]);
  EXPECT_NO_THROW(o4.d_x5[0]);
  EXPECT_NO_THROW(o4.d_x6[0]);
}
TEST(AgradPartialsVari, OperandsAndPartialsFvar) {
  using stan::agrad::OperandsAndPartials;
  using stan::agrad::fvar;

  fvar<double> x1 = 2.0;
  fvar<double> x2 = 3.0;
  fvar<double> x3 = 5.0;
  x1.d_ = 2.0;
  x2.d_ = -1.0;
  x3.d_ = 4.0;

  OperandsAndPartials<fvar<double>,fvar<double>,fvar<double> > o(x1, x2, x3);
  o.d_x1[0] += 17.0; 
  o.d_x2[0] += 19.0;  
  o.d_x3[0] += 23.0;
  fvar<double> y = o.to_var(-1.0,x1,x2,x3);

  EXPECT_FLOAT_EQ(107,y.d_);
  EXPECT_FLOAT_EQ(-1,y.val_);
}

TEST(AgradPartialsVari, incr_deriv_double) {
  using stan::VectorView;
  using stan::agrad::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;

  double a = 1;
  
  VectorView<const double,stan::is_vector<double>::value,
             stan::is_constant_struct<double>::value> d_a(stan::length(a) * 0);


  
  double result = incr_deriv<VectorView<const double,
                                           is_vector<double>::value,
                                           is_constant_struct<double>::value>,
                             double,double>().incr(d_a,a);
  EXPECT_FLOAT_EQ(0, result);
}

TEST(AgradPartialsVari, incr_deriv_vec_double) {
  using stan::VectorView;
  using stan::agrad::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;

  std::vector<double> b;
  b.push_back(1);
  b.push_back(1);
  
  VectorView<const std::vector<double>,
             stan::is_vector<std::vector<double> >::value,
             stan::is_constant_struct<std::vector<double> >::value> 
    d_b(stan::length(b) * 0);

  double result = incr_deriv<VectorView<const std::vector<double>,
                                        is_vector<std::vector<double> >::value,
                                        is_constant_struct<std::vector<double> >::value>,
                             std::vector<double>,double>().incr(d_b,b);

  EXPECT_FLOAT_EQ(0, result);
}


TEST(AgradPartialsVari, incr_deriv_fvar) {
  using stan::VectorView;
  using stan::agrad::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;
  using stan::agrad::fvar;

  fvar<double> c;
  c.val_ = 1.0;
  c.d_ = 1.0;

  double c_deriv = 2;
  VectorView<const double,
             stan::is_vector<fvar<double> >::value,
             stan::is_constant_struct<fvar<double> >::value>
    d_c(c_deriv);

  double result2 = incr_deriv<VectorView<const double,
                                         is_vector<fvar<double> >::value,
                                         is_constant_struct<fvar<double> >::value>,
                              fvar<double>,double>().incr(d_c,c);

  EXPECT_FLOAT_EQ(2, result2);
}

TEST(AgradPartialsVari, incr_deriv_vec_fvar) {
  using stan::VectorView;
  using stan::agrad::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;
  using stan::agrad::fvar;

  fvar<double> c(1,1);

  std::vector<fvar<double> > d;
  d.push_back(c);
  d.push_back(c);
  
  std::vector<double> d_deriv;
  d_deriv.push_back(3);
  d_deriv.push_back(4);

  VectorView<const double,
             stan::is_vector<std::vector<fvar<double> > >::value,
             stan::is_constant_struct<std::vector<fvar<double> > >::value>
    d_d(d_deriv);

  double result3 = incr_deriv<VectorView<const double,
                                         is_vector<std::vector<fvar<double> > >::value,
                                         is_constant_struct<std::vector<fvar<double> > >::value>,
                              std::vector<fvar<double> >,double>().incr(d_d,d);

  EXPECT_FLOAT_EQ(7, result3);
}
