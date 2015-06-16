#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>


TEST(AgradPartialsVari, OperandsAndPartialsFvar) {
  using stan::math::OperandsAndPartials;
  using stan::math::fvar;

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
TEST(AgradPartialsVari, OperandsAndPartialsFvarVec) {
  using stan::math::OperandsAndPartials;
  using stan::math::fvar;

  std::vector<fvar<double> > x1;
  x1.push_back(fvar<double>(2.0,2.0));
  x1.push_back(fvar<double>(1.0,3.0));

  fvar<double> x2 = 3.0;
  fvar<double> x3 = 5.0;
  x2.d_ = -1.0;
  x3.d_ = 4.0;

  OperandsAndPartials<std::vector<fvar<double> >,fvar<double>,fvar<double> > o(x1, x2, x3);
  o.d_x1[0] += 17.0; 
  o.d_x1[1] += 13.0; 
  o.d_x2[0] += 19.0;  
  o.d_x2[0] += 19.0;  
  o.d_x3[0] += 23.0;
  o.d_x3[0] += 23.0;
  fvar<double> y = o.to_var(-1.0,x1,x2,x3);

  EXPECT_FLOAT_EQ(2*17 + 3*13 - 2*19 + 2*4*23,y.d_);
  EXPECT_FLOAT_EQ(-1,y.val_);
}

TEST(AgradPartialsVari, incr_deriv_fvar) {
  using stan::VectorView;
  using stan::math::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;
  using stan::math::fvar;

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
  using stan::math::incr_deriv;
  using stan::is_vector;
  using stan::is_constant_struct;
  using stan::math::fvar;

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
