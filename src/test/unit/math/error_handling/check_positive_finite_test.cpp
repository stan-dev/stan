#include <stan/math/error_handling/check_positive_finite.hpp>
#include <gtest/gtest.h>

using stan::math::check_positive_finite;

TEST(MathErrorHandling,CheckPositiveFinite) {
  const char* function = "check_positive_finite(%1%)";
  double x = 1;
  double result;
 
  EXPECT_TRUE(check_positive_finite(function, x, "x", &result))
    << "check_positive_finite should be true with finite x: " << x;
  x = -1;
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = 0;
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error) 
    << "check_positive_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on NaN: " << x;
}

// ---------- check_positive_finite: vector tests ----------
TEST(MathErrorHandling,CheckPositiveFinite_Vector) {
  const char* function = "check_positive_finite(%1%)";
  double result;
  std::vector<double> x;
  
  x.clear();
  x.push_back (1.5);
  x.push_back (0.1);
  x.push_back (1);
  ASSERT_TRUE(check_positive_finite(function, x, "x", &result)) 
    << "check_positive_finite should be true with finite x";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error) 
    << "check_positive_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error) 
    << "check_positive_finite should throw exception on negative x";

  x.clear();
  x.push_back(0);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error) 
    << "check_positive_finite should throw exception on x=0";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
 << "check_positive_finite should throw exception on NaN";
}

// ---------- check_positive_finite: matrix tests ----------
TEST(MathErrorHandling,CheckPositiveFinite_Matrix) {
  const char* function = "check_positive_finite(%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  result = 0;
  x.resize(3);
  x << 3, 2, 1;
  ASSERT_TRUE(check_positive_finite(function, x, "x", &result))
    << "check_positive_finite should be true with finite x";

  result = 0;
  x.resize(3);
  x << 2, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on Inf";

  result = 0;
  x.resize(3);
  x << 0, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on x=0";

  result = 0;
  x.resize(3);
  x << -1, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on x=-1";

  result = 0;
  x.resize(3);
  x << 2, 1, -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  result = 0;
  x.resize(3);
  x << 1, 2, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, x, "x", &result), std::domain_error) 
    << "check_positive_finite should throw exception on NaN";
}


TEST(MathErrorHandling,CheckPositiveFinite_Matrix_one_indexed_message) {
  const char* function = "check_positive_finite(%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  result = 0;
  x.resize(3);
  x << 1, 2, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, x, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;
}
TEST(MathErrorHandling,CheckPositiveFinite_Matrix_one_indexed_message_2) {
  const char* function = "check_positive_finite(%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  result = 0;
  x.resize(3);
  x << -1, 2, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, x, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[1]"))
    << message;
}

TEST(MathErrorHandling,CheckPositiveFinite_Matrix_one_indexed_message_3) {
  const char* function = "check_positive_finite(%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  result = 0;
  x.resize(3);
  x << 1, 0, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, x, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[2]"))
    << message;
}

