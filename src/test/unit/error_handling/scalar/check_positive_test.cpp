#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/matrix/meta/value_type.hpp>  
#include <gtest/gtest.h>

TEST(MathErrorHandling,CheckPositive_nan) {
  using stan::math::check_positive;
  const char* function = "check_positive(%1%)";
  double result;
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(check_positive(function, nan, "x", &result),
               std::domain_error);

  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(2.0);
  x.push_back(3.0);

  for (int i = 0; i < x.size(); i++) {
    x[i] = nan;
    EXPECT_THROW(check_positive(function, x, "x", &result),
                 std::domain_error);
    x[i] = i;
  }

  Eigen::Matrix<double,Eigen::Dynamic,1> x_mat(3);
  x_mat   << 1, 2, 3;
  for (int i = 0; i < x_mat.size(); i++) {
    x_mat(i) = nan;
    EXPECT_THROW(check_positive(function, x_mat, "x", &result),
                 std::domain_error);
    x_mat(i) = i;
  }
}
