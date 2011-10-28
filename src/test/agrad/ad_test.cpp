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

TEST(agrad_ad,av_times_eq_bv) {
  DEP_FVAR a = 2;
  FVAR b = 3.0;
  a *= b;
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(3.0,a.prime());
}
TEST(agrad_ad,av_times_eq_b) {
  DEP_FVAR a = 2;
  double b = 3.0;
  a *= b;
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(3.0,a.prime());
}


