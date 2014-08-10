#include <stan/math/error_handling/matrix/check_range.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkRange) {

  EXPECT_THROW(stan::math::check_range((size_t)4,(size_t)12, "x",(size_t)4),
               std::out_of_range);

  EXPECT_NO_THROW(stan::math::check_range((size_t)4,(size_t)3, "x",(size_t)4));

  EXPECT_NO_THROW(stan::math::check_range((size_t)4,(size_t)4, "x",(size_t)4));

  EXPECT_NO_THROW(stan::math::check_range((size_t)4,(size_t)1, "x",(size_t)4));
}
