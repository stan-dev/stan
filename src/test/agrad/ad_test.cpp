#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include "stan/agrad/ad.hpp"

typedef stan::agrad::fvar<double> FVAR;
typedef stan::agrad::indep_fvar<double> DEP_FVAR;

TEST(agrad_ad,av_eq) {
  FVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0,a.val());
  EXPECT_FLOAT_EQ(0.0,a.prime());

  DEP_FVAR b = 2.0;
  EXPECT_FLOAT_EQ(2.0,b.val());
  EXPECT_FLOAT_EQ(1.0,b.prime());
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


