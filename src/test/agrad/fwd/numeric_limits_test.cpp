#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, numeric_limits) {
  using stan::agrad::fvar;
  using std::isnan;
  using stan::math::INFTY;

  EXPECT_TRUE(std::numeric_limits<stan::agrad::fvar<double> >::is_specialized);
  EXPECT_TRUE(std::numeric_limits<stan::agrad::fvar<int> >::is_specialized);

  EXPECT_FLOAT_EQ(53, std::numeric_limits<stan::agrad::fvar<double> >::digits);
  EXPECT_FLOAT_EQ(31, std::numeric_limits<stan::agrad::fvar<int> >::digits);

  fvar<int> a = std::numeric_limits<stan::agrad::fvar<int> >::min();
  fvar<double> b = std::numeric_limits<stan::agrad::fvar<double> >::min();
  EXPECT_EQ(-2.147483648e+009, a.val_);
  EXPECT_FLOAT_EQ(2.22507e-308, b.val_);

  fvar<int> c = std::numeric_limits<stan::agrad::fvar<int> >::max();
  fvar<double> d = std::numeric_limits<stan::agrad::fvar<double> >::max();
  EXPECT_FLOAT_EQ(2147483647, c.val_);
  EXPECT_FLOAT_EQ(1.79769e+308, d.val_);

  EXPECT_FLOAT_EQ(15,
                  std::numeric_limits<stan::agrad::fvar<double> >::digits10);
  EXPECT_FLOAT_EQ(9,
                  std::numeric_limits<stan::agrad::fvar<int> >::digits10);

  EXPECT_FLOAT_EQ(1,
                  std::numeric_limits<stan::agrad::fvar<double> >::is_signed);
  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::is_signed);

  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<double> >::is_integer);

  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::is_integer);

  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<double> >::is_exact);

  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::is_exact);
  EXPECT_FLOAT_EQ(2, std::numeric_limits<stan::agrad::fvar<double> >::radix);
  EXPECT_FLOAT_EQ(2, std::numeric_limits<stan::agrad::fvar<int> >::radix);

  fvar<int> e = std::numeric_limits<stan::agrad::fvar<int> >::epsilon();
  fvar<double> f = std::numeric_limits<stan::agrad::fvar<double> >::epsilon();
  EXPECT_FLOAT_EQ(0, e.val_);
  EXPECT_FLOAT_EQ(2.220446e-16, f.val_);

  fvar<int> g = std::numeric_limits<stan::agrad::fvar<int> >::round_error();
  fvar<double> h = std::numeric_limits<stan::agrad::fvar<double> >::round_error();
  EXPECT_FLOAT_EQ(0, g.val_);
  EXPECT_FLOAT_EQ(0.5, h.val_);

  EXPECT_FLOAT_EQ(
      std::numeric_limits<stan::agrad::fvar<double> >::min_exponent, -1021);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::min_exponent);

  EXPECT_FLOAT_EQ(-307,
                std::numeric_limits<stan::agrad::fvar<double> >::min_exponent10);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::min_exponent10);

  EXPECT_FLOAT_EQ(1024,
                  std::numeric_limits<stan::agrad::fvar<double> >::max_exponent);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::max_exponent);

  EXPECT_FLOAT_EQ(308,
               std::numeric_limits<stan::agrad::fvar<double> >::max_exponent10);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::max_exponent10);

  EXPECT_FLOAT_EQ(1,
                  std::numeric_limits<stan::agrad::fvar<double> >::has_infinity);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::has_infinity);

  EXPECT_FLOAT_EQ(1,
                 std::numeric_limits<stan::agrad::fvar<double> >::has_quiet_NaN);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::has_quiet_NaN);

  EXPECT_FLOAT_EQ(1,
             std::numeric_limits<stan::agrad::fvar<double> >::has_signaling_NaN);
  EXPECT_FLOAT_EQ(0,
                std::numeric_limits<stan::agrad::fvar<int> >::has_signaling_NaN);

  EXPECT_FLOAT_EQ(1,
                  std::numeric_limits<stan::agrad::fvar<double> >::has_denorm);
  EXPECT_FLOAT_EQ(0, std::numeric_limits<stan::agrad::fvar<int> >::has_denorm);

  EXPECT_FLOAT_EQ(0,
            std::numeric_limits<stan::agrad::fvar<double> >::has_denorm_loss);
  EXPECT_FLOAT_EQ(0,
                 std::numeric_limits<stan::agrad::fvar<int> >::has_denorm_loss);

  fvar<int> i = std::numeric_limits<stan::agrad::fvar<int> >::infinity();
  fvar<double> j = std::numeric_limits<stan::agrad::fvar<double> >::infinity();
  EXPECT_FLOAT_EQ(0, i.val_);
  EXPECT_FLOAT_EQ(INFTY, j.val_);

  fvar<int> k = std::numeric_limits<stan::agrad::fvar<int> >::quiet_NaN();
  fvar<double> l = 
    std::numeric_limits<stan::agrad::fvar<double> >::quiet_NaN();
  EXPECT_FLOAT_EQ(0, k.val_);
  isnan(l.val_);

  fvar<int> m = std::numeric_limits<stan::agrad::fvar<int> >::signaling_NaN();
  fvar<double> n = 
    std::numeric_limits<stan::agrad::fvar<double> >::signaling_NaN();
  EXPECT_FLOAT_EQ(0, m.val_);
  isnan(n.val_);

  fvar<int> o = std::numeric_limits<stan::agrad::fvar<int> >::denorm_min();
  fvar<double> p = 
    std::numeric_limits<stan::agrad::fvar<double> >::denorm_min();
  EXPECT_FLOAT_EQ(0, o.val_);
  EXPECT_FLOAT_EQ(4.94066e-324, p.val_);

  EXPECT_FLOAT_EQ(1,
                  std::numeric_limits<stan::agrad::fvar<double> >::is_iec559);
  EXPECT_FLOAT_EQ(0, std::numeric_limits<stan::agrad::fvar<int> >::is_iec559);

  EXPECT_FLOAT_EQ(1, 
                  std::numeric_limits<stan::agrad::fvar<double> >::is_bounded);
  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::is_bounded);

  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<double> >::is_modulo);
  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::is_modulo);

  EXPECT_FLOAT_EQ(0, std::numeric_limits<stan::agrad::fvar<double> >::traps);
  EXPECT_FLOAT_EQ(1, std::numeric_limits<stan::agrad::fvar<int> >::traps);

  EXPECT_FLOAT_EQ(0,
              std::numeric_limits<stan::agrad::fvar<double> >::tinyness_before);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::tinyness_before);

  EXPECT_FLOAT_EQ(1,
                  std::numeric_limits<stan::agrad::fvar<double> >::round_style);
  EXPECT_FLOAT_EQ(0,
                  std::numeric_limits<stan::agrad::fvar<int> >::round_style);
}
