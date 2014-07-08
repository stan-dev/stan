#include <stan/math/error_handling/matrix/check_ordered.hpp>
#include <gtest/gtest.h>

using stan::math::check_ordered;

TEST(MathErrorHandlingMatrix, checkOrdered) {
  double result;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y;
  y.resize(3);

  y << 0, 1, 2;
  EXPECT_TRUE(check_ordered("check_ordered(%1%)", y, "y", &result));

  y << 0, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered(%1%)", y, "y", &result));

  y << -10, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered(%1%)", y, "y", &result));

  y << -std::numeric_limits<double>::infinity(), 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered(%1%)", y, "y", &result));

  y << 0, 0, 0;
  EXPECT_THROW(check_ordered("check_ordered(%1%)", y, "y", &result),
               std::domain_error);

  y << 0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_ordered("check_ordered(%1%)", y, "y", &result),
               std::domain_error);


  y << -1, 3, 2;
  EXPECT_THROW(check_ordered("check_ordered(%1%)", y, "y", &result),
               std::domain_error);
}

