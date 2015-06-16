#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/owens_t.hpp>
#include <stan/math/fwd/scal/fun/erf.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdOwensT,Fvar) {
  using stan::math::fvar;
  using stan::math::owens_t;
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

TEST(AgradFwdOwensT,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::owens_t;
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

struct owens_t_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return owens_t(arg1,arg2);
  }
};

TEST(AgradFwdOwensT, nan) {
  owens_t_fun owens_t_;
  test_nan_fwd(owens_t_,3.0,5.0,false);
}
