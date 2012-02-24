#include <gtest/gtest.h>
#include <stan/math/matrix_error_handling.hpp>

TEST(stanMathMatrixErrorHandling, checkNotNanEigenRow) {
  stan::math::vector_d y;
  double result;
  y.resize(3);
  
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
					y, "y", &result));
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
					y, "y"));
  
  y(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow(%1%)",
					 y, "y", &result), std::domain_error);
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow(%1%)",
					 y, "y"), std::domain_error);
  
}
