#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <gtest/gtest.h>

TEST(AgradPartialsVari, incr_deriv_double) {
  using stan::VectorView;
  using stan::math::incr_deriv;
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
  using stan::math::incr_deriv;
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
