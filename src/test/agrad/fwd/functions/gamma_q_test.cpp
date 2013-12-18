#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <stan/agrad/rev/functions/gamma_q.hpp>
#include <stan/agrad/rev.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, gamma_q){
  using stan::agrad::fvar;
  using stan::agrad::gamma_q;
  using boost::math::gamma_q;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  fvar<double> y (1.0);
  y.d_ = 1.0;

  fvar<double> a = gamma_q(x,y);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(0.18228334, a.d_);

  double z = 1.0;
  double w = 0.5;

  a = gamma_q(x,z);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(0.38983709, a.d_);

  a = gamma_q(w,y);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(-boost::math::gamma_p_derivative(0.5,1.0), a.d_);

  EXPECT_THROW(gamma_q(-x,y), std::domain_error);
  EXPECT_THROW(gamma_q(x,-y), std::domain_error);
}
//NEEDS DIGAMMA
// TEST(AgradFvarVar, gamma_q) {
//   using stan::agrad::fvar;
//   using stan::agrad::var;

//   fvar<var> x(0.5,1.0);
//   fvar<var> z(1.0,1.0);
//   fvar<var> a = stan::agrad::gamma_q(x,z);

  // EXPECT_FLOAT_EQ(boost::math::gamma_q(0.5,1.0), a.val_.val());
  // EXPECT_FLOAT_EQ(1.9431838, a.d_.val());

  // AVEC y = createAVEC(x.val_,z.val_);
  // VEC g;
  // a.val_.grad(y,g);
  // EXPECT_FLOAT_EQ(1.7356299,g[0]);
  // EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0),g[1]);
//}
// TEST(AgradFvarFvar, gamma_q) {
//   using stan::agrad::fvar;
//   using boost::math::gamma_q;

//   fvar<fvar<double> > x;
//   x.val_.val_ = 0.5;
//   x.val_.d_ = 1.0;

//   fvar<fvar<double> > y;
//   y.val_.val_ = 1.0;
//   y.d_.val_ = 1.0;

//   fvar<fvar<double> > a = gamma_q(x,y);

//   EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_.val_);
//   EXPECT_FLOAT_EQ(1.7356299, a.val_.d_);
//   EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_);
//   EXPECT_FLOAT_EQ(-0.059628479, a.d_.d_);
// }
