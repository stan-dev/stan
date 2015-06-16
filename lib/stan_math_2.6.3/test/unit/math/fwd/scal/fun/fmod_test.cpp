#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/fmod.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdFmod,Fvar) {
  using stan::math::fvar;
  using std::fmod;
  using std::floor;

  fvar<double> x(2.0,1.0);
  fvar<double> y(3.0,2.0);

  fvar<double> a = fmod(x, y);
  EXPECT_FLOAT_EQ(fmod(2.0, 3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / 3.0), a.d_);

  double z = 4.0;
  double w = 3.0;

  fvar<double> e = fmod(x, z);
  EXPECT_FLOAT_EQ(fmod(2.0, 4.0), e.val_);
  EXPECT_FLOAT_EQ(1.0 / 4.0, e.d_);

  fvar<double> f = fmod(w, x);
  EXPECT_FLOAT_EQ(fmod(3.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(1.0 * -floor(3.0 / 2.0), f.d_);
 }

TEST(AgradFwdFmod,FvarFvarDouble) {
  using stan::math::fvar;
  using std::fmod;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct fmod_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return fmod(arg1,arg2);
  }
};

TEST(AgradFwdFmod, nan) {
  fmod_fun fmod_;
  test_nan_fwd(fmod_,3.0,5.0,false);
}
