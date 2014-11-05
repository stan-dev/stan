#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stdexcept>

TEST(ErrorHandlingMatrix, checkNotNanEigenRow) {
  stan::math::vector_d y;
  y.resize(3);
  
  EXPECT_TRUE(stan::error_handling::check_not_nan("checkNotNanEigenRow(%1)",
                                        "y", y));
  EXPECT_TRUE(stan::error_handling::check_not_nan("checkNotNanEigenRow(%1)",
                                        "y", y));
  
  y(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::error_handling::check_not_nan("checkNotNanEigenRow", "y", y), 
               std::domain_error);
  EXPECT_THROW(stan::error_handling::check_not_nan("checkNotNanEigenRow", "y", y), 
               std::domain_error);
}
