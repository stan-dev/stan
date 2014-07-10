#include <stan/math/error_handling/matrix/check_positive_ordered.hpp>
#include <gtest/gtest.h>

using stan::math::check_positive_ordered;

TEST(MathErrorHandlingMatrix, checkPositiveOrdered) {
  double result;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y;
  y.resize(3);

  y << 0, 1, 2;
  EXPECT_TRUE(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result));

  y << 0, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result));

  y << 0, 0, 0;
  EXPECT_THROW(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result),
               std::domain_error);

  y << 0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result),
               std::domain_error);

  y << -1, 0, 0;
  EXPECT_THROW(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result),
               std::domain_error);

  y << 0, 3, 2;
  EXPECT_THROW(check_positive_ordered("check_positive_ordered(%1%)", y, "y", &result),
               std::domain_error);
}

