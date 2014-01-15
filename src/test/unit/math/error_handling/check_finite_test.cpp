#include <stan/math/error_handling/check_finite.hpp>
#include <gtest/gtest.h>

using stan::math::check_finite;

TEST(MathErrorHandling,CheckFinite) {
  const char* function = "check_finite(%1%)";
  double x = 0;
  double result;
 
  EXPECT_TRUE(check_finite(function, x, "x", &result))
    << "check_finite should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
    << "check_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error) 
    << "check_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
    << "check_finite should throw exception on NaN: " << x;
}

// ---------- check_finite: vector tests ----------
TEST(MathErrorHandling,CheckFinite_Vector) {
  const char* function = "check_finite(%1%)";
  double result;
  std::vector<double> x;
  
  x.clear();
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  ASSERT_TRUE(check_finite(function, x, "x", &result)) 
    << "check_finite should be true with finite x";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error) 
    << "check_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
    << "check_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
 << "check_finite should throw exception on NaN";
}

// ---------- check_finite: matrix tests ----------
TEST(MathErrorHandling,CheckFinite_Matrix) {
  const char* function = "check_finite(%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  result = 0;
  x.resize(3);
  x << -1, 0, 1;
  ASSERT_TRUE(check_finite(function, x, "x", &result))
    << "check_finite should be true with finite x";

  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
    << "check_finite should throw exception on Inf";

  result = 0;
  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error)
    << "check_finite should throw exception on -Inf";
  
  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_finite(function, x, "x", &result), std::domain_error) 
    << "check_finite should throw exception on NaN";
}

