#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar,owens_t) {
  using stan::agrad::fvar;
  using stan::agrad::owens_t;
  using boost::math::owens_t;

  fvar<double> h(1.0,1.0);
  fvar<double> a(2.0,1.0);
  fvar<double> f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(0.0026128467 - 0.1154804963, f.d_);

  f = owens_t(1.0, a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(0.0026128467, f.d_);

  f = owens_t(h, 2.0);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(-0.1154804963, f.d_);
}
TEST(AgradFvarVar,owens_t) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::owens_t;
  using boost::math::owens_t;

  fvar<var> h(1.0,1.0);
  fvar<var> a(2.0,1.0);
  fvar<var> f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val());
  EXPECT_FLOAT_EQ(0.0026128467 - 0.1154804963, f.d_.val());

  AVEC x = createAVEC(h.val_,a.val_);
  VEC grad_f;
  f.val_.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0026128467,grad_f[1]);
  EXPECT_FLOAT_EQ(-0.1154804963,grad_f[0]);
}
TEST(AgradFvarFvar,owens_t) {
  using stan::agrad::fvar;
  using stan::agrad::owens_t;
  using boost::math::owens_t;

  fvar<fvar<double> > h,a;
  h.val_.val_ = 1.0;
  h.val_.d_ = 1.0;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;

  fvar<fvar<double> > f = owens_t(h,a);

  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val_.val_);
  EXPECT_FLOAT_EQ(-0.1154804963, f.val_.d_);
  EXPECT_FLOAT_EQ(0.0026128467, f.d_.val_);
  EXPECT_FLOAT_EQ(-0.013064234,f.d_.d_);
}
