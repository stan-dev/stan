#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include "stan/agrad/ad.hpp"

typedef stan::agrad::fvar<double> FVAR;
typedef stan::agrad::indep_fvar<double> DEP_FVAR;

using stan::agrad::exp;
using std::exp;
using stan::agrad::log;
using std::log;
using stan::agrad::log10;
using std::log10;


TEST(agrad_ad,av_eq) {
  FVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0,a.val());
  EXPECT_FLOAT_EQ(0.0,a.prime());

  DEP_FVAR b = 2.0;
  EXPECT_FLOAT_EQ(2.0,b.val());
  EXPECT_FLOAT_EQ(1.0,b.prime());
}

TEST(agrad_ad,av_plus_eq_bv) {
  DEP_FVAR a = 2.0;
  DEP_FVAR b = 3.0;
  a += b;
  EXPECT_FLOAT_EQ(5.0,a.val());
  EXPECT_FLOAT_EQ(2.0,a.prime());
}
TEST(agrad_ad,av_plus_eq_b) {
  DEP_FVAR a = 2.0;
  double b = 3.0;
  a += b;
  EXPECT_FLOAT_EQ(5.0,a.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
}

TEST(agrad_ad,av_minus_eq_bv) {
  DEP_FVAR a = 2.0;
  DEP_FVAR b = 3.0;
  a -= b;
  EXPECT_FLOAT_EQ(-1.0,a.val());
  EXPECT_FLOAT_EQ(0.0,a.prime());
}
TEST(agrad_ad,av_minus_eq_b) {
  DEP_FVAR a = 2.0;
  double b = 3.0;
  a -= b;
  EXPECT_FLOAT_EQ(-1.0,a.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
}

TEST(agrad_ad,av_div_eq_bv) {
  DEP_FVAR a = 2.0;
  DEP_FVAR b = 3.0;
  a /= b;
  EXPECT_FLOAT_EQ(2.0/3.0,a.val());
  EXPECT_FLOAT_EQ((3.0 - 2.0)/(3.0 * 3.0),a.prime());
}
TEST(agrad_ad,av_div_eq_b) {
  DEP_FVAR a = 2;
  double b = 3.0;
  a /= b;
  EXPECT_FLOAT_EQ(2.0/3.0,a.val());
  EXPECT_FLOAT_EQ(1.0/3.0,a.prime());
}

TEST(agrad_ad,av_eq_bv) {
  FVAR a = 2.0;
  FVAR b = 3.0;
  FVAR c = 2.0;
  EXPECT_TRUE(a == c);
  EXPECT_TRUE(c == a);
  EXPECT_FALSE(a == b);
  EXPECT_FALSE(b == a);
}
TEST(agrad_ad,av_eq_b) {
  FVAR a = 2.0;
  double b = 3.0;
  double c = 2.0;
  EXPECT_TRUE(a == c);
  EXPECT_TRUE(c == a);
  EXPECT_FALSE(a == b);
  EXPECT_FALSE(b == a);
}

TEST(agrad_ad,av_neq_bv) {
  FVAR a = 2.0;
  FVAR b = 3.0;
  FVAR c = 2.0;
  EXPECT_FALSE(a != c);
  EXPECT_FALSE(c != a);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b != a);
}
TEST(agrad_ad,av_neq_b) {
  FVAR a = 2.0;
  double b = 3.0;
  double c = 2.0;
  EXPECT_FALSE(a != c);
  EXPECT_FALSE(c != a);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b != a);
}

TEST(agrad_ad,av_lt_bv) {
  FVAR a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(a < c);
  EXPECT_TRUE(a < d);
}
TEST(agrad_ad,av_lt_b) {
  FVAR a = 2.0;
  double b = 1.0;
  double c = 2.0;
  double d = 3.0;
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(a < c);
  EXPECT_TRUE(a < d);
}
TEST(agrad_ad,a_lt_bv) {
  double a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(a < c);
  EXPECT_TRUE(a < d);
}


TEST(agrad_ad,av_gt_bv) {
  FVAR a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_TRUE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(a > d);
}
TEST(agrad_ad,av_gt_b) {
  FVAR a = 2.0;
  double b = 1.0;
  double c = 2.0;
  double d = 3.0;
  EXPECT_TRUE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(a > d);
}
TEST(agrad_ad,a_gt_bv) {
  double a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_TRUE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(a > d);
}

TEST(agrad_ad,av_lte_bv) {
  FVAR a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_FALSE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(a <= d);
}
TEST(agrad_ad,av_lte_b) {
  FVAR a = 2.0;
  double b = 1.0;
  double c = 2.0;
  double d = 3.0;
  EXPECT_FALSE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(a <= d);
}
TEST(agrad_ad,a_lte_bv) {
  double a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_FALSE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(a <= d);
}


TEST(agrad_ad,av_gte_bv) {
  FVAR a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= d);
}
TEST(agrad_ad,av_gte_b) {
  FVAR a = 2.0;
  double b = 1.0;
  double c = 2.0;
  double d = 3.0;
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= d);
}
TEST(agrad_ad,a_gte_bv) {
  double a = 2.0;
  FVAR b = 1.0;
  FVAR c = 2.0;
  FVAR d = 3.0;
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a >= c);
  EXPECT_FALSE(a >= d);
}

TEST(agrad_ad,negation_a) {
  FVAR a = -1.0;
  FVAR b = 0.0;
  FVAR c = 1.0;
  EXPECT_FALSE(!a);
  EXPECT_TRUE(!b);
  EXPECT_FALSE(!c);
}

TEST(agrad_ad,pos_a) {
  DEP_FVAR a = -1.0;
  DEP_FVAR b = 0.0;
  DEP_FVAR c = 1.0;
  FVAR p_a = +a;
  FVAR p_b = +b;
  FVAR p_c = +c;
  EXPECT_FLOAT_EQ(-1.0,p_a.val());
  EXPECT_FLOAT_EQ(0.0,p_b.val());
  EXPECT_FLOAT_EQ(1.0,p_c.val());
  EXPECT_FLOAT_EQ(1.0,p_a.prime());
  EXPECT_FLOAT_EQ(1.0,p_a.prime());
  EXPECT_FLOAT_EQ(1.0,p_a.prime());
}

TEST(agrad_ad,negative_a) {
  DEP_FVAR a = -1.0;
  DEP_FVAR b = 0.0;
  DEP_FVAR c = 1.0;
  FVAR p_a = -a;
  FVAR p_b = -b;
  FVAR p_c = -c;
  EXPECT_FLOAT_EQ(1.0,p_a.val());
  EXPECT_FLOAT_EQ(0.0,p_b.val());
  EXPECT_FLOAT_EQ(-1.0,p_c.val());
  EXPECT_FLOAT_EQ(-1.0,p_a.prime());
  EXPECT_FLOAT_EQ(-1.0,p_a.prime());
  EXPECT_FLOAT_EQ(-1.0,p_a.prime());
}

TEST(agrad_ad,a_plus_b) {
  DEP_FVAR a = 2.0;
  FVAR b = 3.0;
  FVAR c = a + b;
  EXPECT_FLOAT_EQ(5.0, c.val());
  EXPECT_FLOAT_EQ(1.0, c.prime());

  DEP_FVAR d = 4.0;
  FVAR e = d + d;
  EXPECT_FLOAT_EQ(8.0,e.val());
  EXPECT_FLOAT_EQ(2.0,e.prime());

  DEP_FVAR f = 5.0;
  double g = 2.0;
  FVAR h = f + g;
  EXPECT_FLOAT_EQ(7.0,h.val());
  EXPECT_FLOAT_EQ(1.0,h.prime());
  
  double i = 7.0;
  DEP_FVAR j = 11.0;
  FVAR k = i + j;
  EXPECT_FLOAT_EQ(18.0,k.val());
  EXPECT_FLOAT_EQ(1.0,k.prime());
}

TEST(agrad_ad,av_minus_bv) {
  DEP_FVAR a = 2;
  FVAR b = 3.0;
  FVAR f = a - b;
  EXPECT_FLOAT_EQ(-1.0,f.val());
  EXPECT_FLOAT_EQ(1.0,f.prime());
}
TEST(agrad_ad,av_minus_b) {
  DEP_FVAR a = 2;
  double b = 3.0;
  FVAR f = a - b;
  EXPECT_FLOAT_EQ(-1.0,f.val());
  EXPECT_FLOAT_EQ(1.0,f.prime());
}
TEST(agrad_ad,a_minus_bv) {
  double a = 2.0;
  DEP_FVAR b = 3.0;
  FVAR f = a - b;
  EXPECT_FLOAT_EQ(-1.0,f.val());
  EXPECT_FLOAT_EQ(-1.0,f.prime());
}

TEST(agrad_ad,av_times_bv) {
  DEP_FVAR a = 2;
  FVAR b = 3.0;
  FVAR f = a * b;
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(3.0,f.prime());
}
TEST(agrad_ad,av_times_b) {
  DEP_FVAR a = 2;
  double b = 3.0;
  FVAR f = a * b;
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(3.0,f.prime());
}
TEST(agrad_ad,a_times_bv) {
  double a = 2.0;
  DEP_FVAR b = 3.0;
  FVAR f = a * b;
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(2.0,f.prime());
}

TEST(agrad_ad,av_div_bv) {
  DEP_FVAR a = 2.0;
  FVAR b = 3.0;
  FVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0/3.0, f.val());
  EXPECT_FLOAT_EQ(3.0 / (3.0 * 3.0), f.prime());
}
TEST(agrad_ad,av_div_bv_2) {
  FVAR c = 2.0;
  DEP_FVAR d = 3.0;
  FVAR g = c / d;
  EXPECT_FLOAT_EQ(2.0/3.0, g.val());
  EXPECT_FLOAT_EQ(-2.0/(3.0 * 3.0), g.prime());
}
TEST(agrad_ad,av_div_bv_3) {
  DEP_FVAR c = 2.0;
  DEP_FVAR d = 3.0;
  FVAR g = c / d;
  EXPECT_FLOAT_EQ(2.0/3.0, g.val());
  EXPECT_FLOAT_EQ((3.0 -2.0)/(3.0 * 3.0), g.prime());
}

TEST(agrad_ad,av_div_b) {
  DEP_FVAR a = 2.0;
  double b = 3.0;
  FVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0/3.0, f.val());
  EXPECT_FLOAT_EQ(3.0 / (3.0 * 3.0), f.prime());
}
TEST(agrad_ad,a_div_bv_2) {
  double c = 2.0;
  DEP_FVAR d = 3.0;
  FVAR g = c / d;
  EXPECT_FLOAT_EQ(2.0/3.0, g.val());
  EXPECT_FLOAT_EQ(-2.0/(3.0 * 3.0), g.prime());
}

TEST(agrad_ad,av_plus_plus) {
  DEP_FVAR a = -2.0;
  FVAR b = a++;
  EXPECT_FLOAT_EQ(-1.0,a.val());
  EXPECT_FLOAT_EQ(-2.0,b.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
  EXPECT_FLOAT_EQ(1.0,b.prime());
}

TEST(agrad_ad,plus_plus_av) {
  DEP_FVAR a = -2.0;
  FVAR b = ++a;
  EXPECT_FLOAT_EQ(-1.0,a.val());
  EXPECT_FLOAT_EQ(-1.0,b.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
  EXPECT_FLOAT_EQ(1.0,b.prime());
}

TEST(agrad_ad,av_minus_minus) {
  DEP_FVAR a = -2.0;
  FVAR b = a--;
  EXPECT_FLOAT_EQ(-3.0,a.val());
  EXPECT_FLOAT_EQ(-2.0,b.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
  EXPECT_FLOAT_EQ(1.0,b.prime());
}

TEST(agrad_ad,minus_minus_av) {
  DEP_FVAR a = -2.0;
  FVAR b = --a;
  EXPECT_FLOAT_EQ(-3.0,a.val());
  EXPECT_FLOAT_EQ(-3.0,b.val());
  EXPECT_FLOAT_EQ(1.0,a.prime());
  EXPECT_FLOAT_EQ(1.0,b.prime());
}

TEST(agrad_ad,exp_av) {
  DEP_FVAR a = -2.0;
  FVAR b = exp(a);
  EXPECT_FLOAT_EQ(exp(-2.0), b.val());
  EXPECT_FLOAT_EQ(exp(-2.0), b.prime());

  DEP_FVAR c = 3.0;
  FVAR d = exp(2 * c);
  EXPECT_FLOAT_EQ(exp(6.0), d.val());
  EXPECT_FLOAT_EQ(2.0 * exp(6.0), d.prime());
}


TEST(agrad_ad,log_av) {
  DEP_FVAR a = 3.0;
  FVAR b = log(a);
  EXPECT_FLOAT_EQ(log(3.0), b.val());
  EXPECT_FLOAT_EQ(1.0/3.0, b.prime());
}
TEST(agrad_ad,log_av_2) {
  DEP_FVAR c = 3.0;
  FVAR d = log(2 * c);
  EXPECT_FLOAT_EQ(log(6.0), d.val());
  EXPECT_FLOAT_EQ(1.0/3.0, d.prime());
}

TEST(agrad_ad,log10_av) {
  DEP_FVAR a = 3.0;
  FVAR b = log10(a);
  EXPECT_FLOAT_EQ(log10(3.0), b.val());
  EXPECT_FLOAT_EQ(1.0/3.0/log(10.0), b.prime());
}
TEST(agrad_ad,log10_av_2) {
  DEP_FVAR c = 3.0;
  FVAR d = log10(2 * c);
  EXPECT_FLOAT_EQ(log10(6.0), d.val());
  EXPECT_FLOAT_EQ(1.0/3.0/log(10.0), d.prime());
}
