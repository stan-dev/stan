#include <gtest/gtest.h>
#include <stan/prob/internal_math.hpp>

TEST(ProbInternalMath, gradRegIncGamma_typical) {
  double a = 0.5;
  double b = 1.0;
  double g = 1.77245;
  double dig = -1.96351;
  
  EXPECT_FLOAT_EQ(0.38984156, stan::math::gradRegIncGamma(a, b, g, dig));
}

TEST(ProbInternalMath, gradRegIncGamma_infLoopInVersion2_0_1) {
  double a = 8.01006;
  double b = 2.47579e+215;
  double g = 5143.28;
  double dig = 2.01698;
  
  EXPECT_THROW(stan::math::gradRegIncGamma(a, b, g, dig),
               std::domain_error);
}
