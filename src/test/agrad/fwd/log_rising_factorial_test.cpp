#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(log_rising_factorial,AgradFvar) {
  using stan::agrad::fvar;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = log_rising_factorial(a,1.0);
  EXPECT_FLOAT_EQ(std::log(4.0), x.val_);
  EXPECT_FLOAT_EQ(0.25, x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(log_rising_factorial(c, 2), std::domain_error);
  EXPECT_THROW(log_rising_factorial(2, c), std::domain_error);
  EXPECT_THROW(log_rising_factorial(c, c), std::domain_error);

  x = log_rising_factorial(a,a);
  EXPECT_FLOAT_EQ(std::log(840.0), x.val_);
  EXPECT_FLOAT_EQ((2 * digamma(8) - digamma(4)), x.d_);

  x = log_rising_factorial(5, a);
  EXPECT_FLOAT_EQ(std::log(1680.0), x.val_);
  EXPECT_FLOAT_EQ(digamma(9), x.d_);
}
TEST(log_rising_factorial,AgradFvarVar_FvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(2.4894509, c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
  EXPECT_FLOAT_EQ(1.8727844, g[1]);
}
TEST(log_rising_factorial,AgradFvarVar_Double_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(3.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, c.d_.val());

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
}
TEST(log_rising_factorial,AgradDouble_FvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(1.8727844, c.d_.val());

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.8727844, g[0]);
}

TEST(log_rising_factorial,AgradFvarVar_FvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.023267401, g[0]);
  EXPECT_FLOAT_EQ(0.30709034, g[1]);
}
TEST(log_rising_factorial,AgradFvarVar_Double_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(3.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
}
TEST(log_rising_factorial,AgradDouble_FvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
}
TEST(log_rising_factorial,AgradFvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_);
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_);
  EXPECT_FLOAT_EQ(0.15354517, a.d_.d_);
}
TEST(log_rising_factorial,AgradFvarFvarVar_FvarFvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.15354517, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
  EXPECT_FLOAT_EQ(1.8727844, g[1]);
}
TEST(log_rising_factorial,AgradFvarFvarVar_Double_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(3.0);

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
}
TEST(log_rising_factorial,AgradDouble_FvarFvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.8727844, g[0]);
}
TEST(log_rising_factorial,AgradFvarFvarVar_FvarFvarVar_2ndderiv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
  EXPECT_FLOAT_EQ(0.15354517, g[1]);
}
TEST(log_rising_factorial,AgradFvarFvarVar_FvarFvarVar_2ndderiv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
  EXPECT_FLOAT_EQ(0.15354517, g[1]);
}
TEST(log_rising_factorial,AgradFvarFvarVar_Double_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(3.0);

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
}
TEST(log_rising_factorial,AgradDouble_FvarFvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
}
