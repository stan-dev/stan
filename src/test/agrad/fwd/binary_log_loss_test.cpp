#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/binary_log_loss.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::math::binary_log_loss;
  using std::log;
  using std::isnan;

  fvar<double> w(0.0,1.0);
  fvar<double> x(1.0,2.0);
  fvar<double> y(0.4,3.0);

  fvar<double> a = binary_log_loss(w, y);
  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_);
  EXPECT_FLOAT_EQ(-1.0 * log(0.4) + 1.0 * log(1 - 0.4) + 3.0 / 0.6, a.d_);

  fvar<double> b = binary_log_loss(x, y);
  EXPECT_FLOAT_EQ(binary_log_loss(1.0, 0.4), b.val_);
  EXPECT_FLOAT_EQ(-2.0 * log(0.4) + 2.0 * log(1 - 0.4) - 3.0 * 1.0 / 0.4, b.d_);
}

// TEST(AgradFvarVar, binary_log_loss) {
//   using stan::agrad::fvar;
//   using stan::agrad::var;
//   using stan::math::binary_log_loss;

//   fvar<var> x;
//   x.val_ = 0.0;
//   x.d_ = 1.0;

//   fvar<var> z;
//   z.val_ = 0.4;
//   z.d_ = 3.0;
//   fvar<var> a = binary_log_loss(x,z);

//   EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_.val());
//   EXPECT_FLOAT_EQ(-1.0 * log(0.4) + 1.0 * log(1 - 0.4) + 3.0 / 0.6, a.d_.val());

//   AVEC y = createAVEC(x.val_);
//   VEC g;
//   a.val_.grad(y,g);
//   EXPECT_FLOAT_EQ(0, g[0]);
//  EXPECT_FLOAT_EQ(0, g[1]);

//   y = createAVEC(x.d_);
//   a.d_.grad(y,g);
//   EXPECT_FLOAT_EQ(0, g[0]);
//  EXPECT_FLOAT_EQ(0, g[1]);
// }

TEST(AgradFvarFvar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::math::binary_log_loss;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = binary_log_loss(x,y);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0,0.4), a.val_.val_);
  EXPECT_FLOAT_EQ(-1.0 * log(0.4) + 1.0 * log(0.6), a.val_.d_);
  EXPECT_FLOAT_EQ(1.0 * 1 / (1 - 0.4), a.d_.val_);
  EXPECT_FLOAT_EQ(-25.0 / 6.0, a.d_.d_);
}
