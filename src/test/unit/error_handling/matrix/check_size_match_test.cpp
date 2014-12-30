#include <stan/error_handling/matrix/check_size_match.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkSizeMatch) {
  using stan::error_handling::check_size_match;
  int x;
  size_t y;

  x = 3;
  y = 4;
  EXPECT_THROW(check_size_match("checkSizeMatch", "x", x, "y", y),
               std::invalid_argument);
  EXPECT_THROW(check_size_match("checkSizeMatch", "y", y, "x", x),
               std::invalid_argument);

  x = 2;
  y = 2;
  EXPECT_TRUE(check_size_match("checkSizeMatch", "x", x, "y", y));
  EXPECT_TRUE(check_size_match("checkSizeMatch", "y", y, "x", x));
}
