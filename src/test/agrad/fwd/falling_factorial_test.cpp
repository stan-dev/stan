#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, falling_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::falling_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = falling_factorial(a,1);
  EXPECT_FLOAT_EQ(24.0, x.val_);
  EXPECT_FLOAT_EQ(24.0 * digamma(5), x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(falling_factorial(c, 2), std::domain_error);
  EXPECT_THROW(falling_factorial(2, c), std::domain_error);
  EXPECT_THROW(falling_factorial(c, c), std::domain_error);

  x = falling_factorial(a,a);
  EXPECT_FLOAT_EQ(1.0, x.val_);
  EXPECT_FLOAT_EQ(0.0, x.d_);

  x = falling_factorial(5, a);
  EXPECT_FLOAT_EQ(5.0, x.val_);
  EXPECT_FLOAT_EQ(-5.0 * digamma(5.0),x.d_);
}
//WONT WORK UNTIL STUFF IN CLEAN_UP_FVAR IS MERGED IN -- REQUIRES DIGAMMA FUNCTION
// TEST(AgradFvarVar, falling_factorial) {
//   using stan::agrad::fvar;
//   using stan::agrad::var;
//   using stan::agrad::falling_factorial;

//   fvar<var> a(4.0,1.0);
//   fvar<var> b(4.0,1.0);
//   fvar<var> c = falling_factorial(a,b);

//   EXPECT_FLOAT_EQ(1, c.val_.val());
//   EXPECT_FLOAT_EQ(0, c.d_.val());

//   AVEC y = createAVEC(a.val_,b.val_);
//   VEC g;
//   c.val_.grad(y,g);
//   EXPECT_FLOAT_EQ(0, g[0]);
//   EXPECT_FLOAT_EQ(0, g[1]);
// }

// TEST(AgradFvarFvar, falling_factorial) {
//   using stan::agrad::fvar;
//   using stan::math::falling_factorial;

//   fvar<fvar<double> > x;
//   x.val_.val_ = 4.0;
//   x.val_.d_ = 1.0;

//   fvar<fvar<double> > y;
//   y.val_.val_ = 4.0;
//   y.d_.val_ = 1.0;

//   fvar<fvar<double> > a = falling_factorial(x,y);

//   EXPECT_FLOAT_EQ(falling_factorial(4,4.0), a.val_.val_);
//   EXPECT_FLOAT_EQ(0, a.val_.d_);
//   EXPECT_FLOAT_EQ(0, a.d_.val_);
//   EXPECT_FLOAT_EQ(0, a.d_.d_);
// }
