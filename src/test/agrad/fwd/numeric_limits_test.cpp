#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, numeric_limits) {
  using stan::agrad::fvar;
  using std::isnan;
  using stan::math::INFTY;

  EXPECT_TRUE(std::numeric_limits<stan::agrad::fvar<double> >::is_specialized);
  EXPECT_TRUE(std::numeric_limits<stan::agrad::fvar<int> >::is_specialized);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::digits, 53);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::digits, 31);

  fvar<int> a = std::numeric_limits<stan::agrad::fvar<int> >::min();
  fvar<double> b = std::numeric_limits<stan::agrad::fvar<double> >::min();
  EXPECT_FLOAT_EQ(a.val_, -2147483648);
  EXPECT_FLOAT_EQ(b.val_, 2.22507e-308);

  fvar<int> c = std::numeric_limits<stan::agrad::fvar<int> >::max();
  fvar<double> d = std::numeric_limits<stan::agrad::fvar<double> >::max();
  EXPECT_FLOAT_EQ(c.val_, 2147483647);
  EXPECT_FLOAT_EQ(d.val_, 1.79769e+308);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::digits10, 
                  15);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::digits10, 9);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_signed, 
                  1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_signed, 1);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_integer,
                  0);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_integer, 1);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_exact, 
                  0);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_exact, 1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::radix, 2);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::radix, 2);

  fvar<int> e = std::numeric_limits<stan::agrad::fvar<int> >::epsilon();
  fvar<double> f = std::numeric_limits<stan::agrad::fvar<double> >::epsilon();
  EXPECT_FLOAT_EQ(e.val_, 0);
  EXPECT_FLOAT_EQ(f.val_, 2.220446e-16);

  fvar<int> g = std::numeric_limits<stan::agrad::fvar<int> >::round_error();
  fvar<double> h = std::numeric_limits<stan::agrad::fvar<double> >::round_error();
  EXPECT_FLOAT_EQ(g.val_, 0);
  EXPECT_FLOAT_EQ(h.val_,0.5);

  EXPECT_FLOAT_EQ(
      std::numeric_limits<stan::agrad::fvar<double> >::min_exponent, -1021);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::min_exponent, 
                  0);

  EXPECT_FLOAT_EQ(
        std::numeric_limits<stan::agrad::fvar<double> >::min_exponent10, -307);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::min_exponent10,
                  0);

  EXPECT_FLOAT_EQ(
          std::numeric_limits<stan::agrad::fvar<double> >::max_exponent, 1024);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::max_exponent,
                  0);

  EXPECT_FLOAT_EQ(
         std::numeric_limits<stan::agrad::fvar<double> >::max_exponent10, 308);
  EXPECT_FLOAT_EQ(
              std::numeric_limits<stan::agrad::fvar<int> >::max_exponent10, 0);

  EXPECT_FLOAT_EQ(
             std::numeric_limits<stan::agrad::fvar<double> >::has_infinity, 1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::has_infinity, 
                  0);

  EXPECT_FLOAT_EQ(
            std::numeric_limits<stan::agrad::fvar<double> >::has_quiet_NaN, 1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::has_quiet_NaN,
                  0);

  EXPECT_FLOAT_EQ(
        std::numeric_limits<stan::agrad::fvar<double> >::has_signaling_NaN, 1);
  EXPECT_FLOAT_EQ(
           std::numeric_limits<stan::agrad::fvar<int> >::has_signaling_NaN, 0);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::has_denorm,
                  1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::has_denorm, 0);

  EXPECT_FLOAT_EQ(
          std::numeric_limits<stan::agrad::fvar<double> >::has_denorm_loss, 0);
  EXPECT_FLOAT_EQ(
             std::numeric_limits<stan::agrad::fvar<int> >::has_denorm_loss, 0);

  fvar<int> i = std::numeric_limits<stan::agrad::fvar<int> >::infinity();
  fvar<double> j = std::numeric_limits<stan::agrad::fvar<double> >::infinity();
  EXPECT_FLOAT_EQ(i.val_, 0);
  EXPECT_FLOAT_EQ(j.val_, INFTY);

  fvar<int> k = std::numeric_limits<stan::agrad::fvar<int> >::quiet_NaN();
  fvar<double> l = 
    std::numeric_limits<stan::agrad::fvar<double> >::quiet_NaN();
  EXPECT_FLOAT_EQ(k.val_, 0);
  isnan(l.val_);

  fvar<int> m = std::numeric_limits<stan::agrad::fvar<int> >::signaling_NaN();
  fvar<double> n = 
    std::numeric_limits<stan::agrad::fvar<double> >::signaling_NaN();
  EXPECT_FLOAT_EQ(m.val_, 0);
  isnan(n.val_);

  fvar<int> o = std::numeric_limits<stan::agrad::fvar<int> >::denorm_min();
  fvar<double> p = 
    std::numeric_limits<stan::agrad::fvar<double> >::denorm_min();
  EXPECT_FLOAT_EQ(o.val_, 0);
  EXPECT_FLOAT_EQ(p.val_, 4.94066e-324);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_iec559, 
                  1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_iec559, 0);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_bounded,
                  1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_bounded, 1);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::is_modulo, 
                  0);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::is_modulo, 1);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::traps, 0);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::traps, 1);

  EXPECT_FLOAT_EQ(
          std::numeric_limits<stan::agrad::fvar<double> >::tinyness_before, 0);
  EXPECT_FLOAT_EQ(
             std::numeric_limits<stan::agrad::fvar<int> >::tinyness_before, 0);

  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<double> >::round_style,
                  1);
  EXPECT_FLOAT_EQ(std::numeric_limits<stan::agrad::fvar<int> >::round_style, 
                  0);
}
