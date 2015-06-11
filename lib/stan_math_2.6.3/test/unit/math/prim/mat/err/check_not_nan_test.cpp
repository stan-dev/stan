#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stdexcept>

TEST(ErrorHandlingMatrix, checkNotNanEigenRow) {
  stan::math::vector_d y;
  y.resize(3);
  
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
                                        "y", y));
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
                                        "y", y));
  
  y(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow", "y", y), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow", "y", y), 
               std::domain_error);
}
