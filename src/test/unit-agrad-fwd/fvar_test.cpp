#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>

TEST(AgradFwdFvar,Fvar) {
  using stan::agrad::fvar;
  typedef stan::agrad::fvar<double> fvd;

  fvar<double> a;
  EXPECT_FLOAT_EQ(0.0, a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b(1.9);
  EXPECT_FLOAT_EQ(1.9, b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c(1.93, -27.832);
  EXPECT_FLOAT_EQ(1.93, c.val_);
  EXPECT_FLOAT_EQ(-27.832, c.d_);

  fvar<double> d = -c;
  EXPECT_FLOAT_EQ(-1.93, d.val_);
  EXPECT_FLOAT_EQ(27.832, d.d_);

  fvar<double> e(5.0);
  d += e;
  EXPECT_FLOAT_EQ(3.07, d.val_);

  EXPECT_FLOAT_EQ(3.07, (d++).val_);
  EXPECT_FLOAT_EQ(4.07, d.val_);

  EXPECT_FLOAT_EQ(5.07, (++d).val_);
  EXPECT_FLOAT_EQ(5.07, d.val_);

  double nan = std::numeric_limits<double>::quiet_NaN();
  fvar<double> f(nan);
  EXPECT_TRUE(boost::math::isnan(f.val_));
  EXPECT_TRUE(boost::math::isnan(f.d_));
  
  fvar<double> g(nan, 1);
  EXPECT_TRUE(boost::math::isnan(g.val_));
  EXPECT_TRUE(boost::math::isnan(g.d_));

  fvar<double> h(nan, nan);
  EXPECT_TRUE(boost::math::isnan(h.val_));
  EXPECT_TRUE(boost::math::isnan(h.d_));

  fvar<double> i(4, nan);
  EXPECT_FLOAT_EQ(4, i.val_);
  EXPECT_TRUE(boost::math::isnan(i.d_));
}
