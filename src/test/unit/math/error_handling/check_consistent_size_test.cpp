#include <stan/math/error_handling/check_consistent_size.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixErrorHandling, checkConsistentSize) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize(%1%)";
  const char* name1 = "name1";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(4U, function, v1, name1, &result));
  EXPECT_THROW(check_consistent_size(2U, function, v1, name1, &result), std::domain_error);
}

TEST(MathMatrixErrorHandling, checkConsistentSize_message) {
  
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize(%1%)";
  const char* name1 = "name1";
  
  double result;
  std::string message;
  Matrix<double,Dynamic,1> v1(4);
  EXPECT_EQ(4U, size_of(v1));
  
  try {
    check_consistent_size(2U, function, v1, name1, &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_TRUE(std::string::npos != message.find("name1 dimension=4"))
    << message;
  EXPECT_TRUE(std::string::npos != message.find("1 or max_size=2"))
    << message;
}
