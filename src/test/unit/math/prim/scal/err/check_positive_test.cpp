#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>  
#include <gtest/gtest.h>

TEST(ErrorHandlingScalar,CheckPositive) {
  using stan::math::check_positive;
  const char* function = "check_positive";

  EXPECT_TRUE(check_positive(function, "x", nan));

  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(2.0);
  x.push_back(3.0);

  for (size_t i = 0; i < x.size(); i++) {
    EXPECT_TRUE(check_positive(function, "x", x));
  }

  Eigen::Matrix<double,Eigen::Dynamic,1> x_mat(3);
  x_mat   << 1, 2, 3;
  for (int i = 0; i < x_mat.size(); i++) {
    EXPECT_TRUE(check_positive(function, "x", x_mat));
  }

  x_mat(0) = 0;

  EXPECT_THROW(check_positive(function, "x", x_mat),
               std::domain_error);
}

TEST(ErrorHandlingScalar,CheckPositive_nan) {
  using stan::math::check_positive;
  const char* function = "check_positive";

  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(check_positive(function, "x", nan),
               std::domain_error);

  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(2.0);
  x.push_back(3.0);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = nan;
    EXPECT_THROW(check_positive(function, "x", x),
                 std::domain_error);
    x[i] = i;
  }

  Eigen::Matrix<double,Eigen::Dynamic,1> x_mat(3);
  x_mat   << 1, 2, 3;
  for (int i = 0; i < x_mat.size(); i++) {
    x_mat(i) = nan;
    EXPECT_THROW(check_positive(function, "x", x_mat),
                 std::domain_error);
    x_mat(i) = i;
  }
}
