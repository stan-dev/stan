#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/falling_factorial.hpp>
#include <stan/math/rev/scal/fun/falling_factorial.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>


TEST(AgradFwdFallingFactorial,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = falling_factorial(a,b);

  EXPECT_FLOAT_EQ(1, c.val_.val());
  EXPECT_FLOAT_EQ(0, c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.5061177, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = falling_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
  EXPECT_FLOAT_EQ(-0.22132295, g[1]);
}

TEST(AgradFwdFallingFactorial,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  EXPECT_FLOAT_EQ(falling_factorial(4,4.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.5061177, a.d_.val_.val());
  EXPECT_FLOAT_EQ(-2.2683904, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.5061177, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.4897134, g[0]);
  EXPECT_FLOAT_EQ(-2.2683904, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-2.2683904, g[0]);
  EXPECT_FLOAT_EQ(2.0470674, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-3.7498014, g[0]);
  EXPECT_FLOAT_EQ(3.0831244, g[1]);
}

struct falling_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return falling_factorial(arg1,arg2);
  }
};

TEST(AgradFwdFallingFactorial, nan) {
  falling_factorial_fun falling_factorial_;
  test_nan_mix(falling_factorial_,3.0,5.0,false);
}
