#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(log_sum_exp,AgradFvar) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);
  double z = 1.4;

  fvar<double> a = log_sum_exp(x, y);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ((1.0 * exp(0.5) + 2.0 * exp(1.2)) / (exp(0.5) 
                                                       + exp(1.2)), a.d_);

  fvar<double> b = log_sum_exp(x, z);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.4), b.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), b.d_);

  fvar<double> c = log_sum_exp(z, x);
  EXPECT_FLOAT_EQ(log_sum_exp(1.4, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), c.d_);
}
TEST(log_sum_exp,AgradFvarVar_FvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(log_sum_exp,AgradFvarVar_Double_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(log_sum_exp,AgradDouble_FvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(log_sum_exp,AgradFvarVar_FvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());


  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (1.3* exp(3.0) + exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (1.3* exp(3.0) + exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(log_sum_exp,AgradFvarVar_Double_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = log_sum_exp(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * (exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) * exp(3.0))
                  / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(log_sum_exp,AgradDouble_FvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) * exp(6.0))
                  / (exp(3.0) + exp(6.0)) / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(log_sum_exp,AgradFvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_);
}
TEST(log_sum_exp,AgradFvarFvarVar_FvarFvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[1]);
}
TEST(log_sum_exp,AgradFvarFvarVar_Double_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(log_sum_exp,AgradDouble_FvarFvarVar_1stderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), g[0]);
}
TEST(log_sum_exp,AgradFvarFvarVar_FvarFvarVar_2ndderiv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (exp(3.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(-0.045176659, g[1]);
}
TEST(log_sum_exp,AgradFvarFvarVar_FvarFvarVar_2ndderiv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.045176659, g[0]);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[1]);
}
TEST(log_sum_exp,AgradFvarFvarVar_Double_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(3.0) * (exp(3.0) + exp(6.0)) - exp(3.0) 
                   * (exp(3.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
}
TEST(log_sum_exp,AgradDouble_FvarFvarVar_2ndderiv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_sum_exp(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ((exp(6.0) * (exp(3.0) + exp(6.0)) - exp(6.0) 
                   * (exp(6.0))) / (exp(3.0) + exp(6.0)) 
                  / (exp(3.0) + exp(6.0)),g[0]);
}
