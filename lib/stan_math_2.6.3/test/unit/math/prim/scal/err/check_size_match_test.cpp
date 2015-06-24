#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingMatrix, checkSizeMatch) {
  using stan::math::check_size_match;
  int x;
  size_t y;

  x = 3;
  y = 4;
  EXPECT_THROW_MSG(check_size_match("checkSizeMatch", "x", x, 
                                    "y", y),
                   std::invalid_argument,
                   "x (3) and y (4) must match in size");
  EXPECT_THROW_MSG(check_size_match("checkSizeMatch", "expr_x ", "x", x, 
                                    "expr_y ", "y", y),
                   std::invalid_argument,
                   "expr_x x (3) and expr_y y (4) must match in size");

  EXPECT_THROW_MSG(check_size_match("checkSizeMatch", "y", y, 
                                    "x", x),
                   std::invalid_argument,
                   "y (4) and x (3) must match in size");
  EXPECT_THROW_MSG(check_size_match("checkSizeMatch", "expr_y ", "y", y, 
                                    "expr_x ", "x", x),
                   std::invalid_argument,
                   "expr_y y (4) and expr_x x (3) must match in size");

  x = 2;
  y = 2;
  EXPECT_TRUE(check_size_match("checkSizeMatch", "x", x, "y", y));
  EXPECT_TRUE(check_size_match("checkSizeMatch", "expr_x ", "x", x, "expr_y ", "y", y));
  EXPECT_TRUE(check_size_match("checkSizeMatch", "y", y, "x", x));
  EXPECT_TRUE(check_size_match("checkSizeMatch", "expr_y ", "y", y, "expr_x", "x", x));
}

