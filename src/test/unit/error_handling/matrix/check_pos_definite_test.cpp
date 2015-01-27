#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>


const char* function = "function";
class ErrorHandlingMatrix : public ::testing::Test {
public:
  void SetUp() {
  }
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
};

TEST_F(ErrorHandlingMatrix, checkPosDefinite) {
  using stan::math::check_pos_definite;

  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_pos_definite(function, "y", y));

  y.resize(3,3);
  y << 
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;
  EXPECT_TRUE(check_pos_definite(function, "y", y));
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_not_square) {
  using stan::math::check_pos_definite;
  std::string expected_msg;
  
  y.resize(3, 4);
  expected_msg = "Expecting a square matrix; rows of y (3) and columns of y (4) must match in size";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::invalid_argument,
                   expected_msg);
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_0_size) {
  using stan::math::check_pos_definite;
  std::string expected_msg;

  expected_msg = "y must have a positive size, but is 0; dimension size expression = rows";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::invalid_argument,
                   expected_msg);
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_non_symmetric) {
  using stan::math::check_pos_definite;
  std::string expected_msg;

  y.resize(3,3);
  y <<
    1, 0, 0,
    0, 1, 0.5,
    0, 0, 1;
  
  expected_msg = "y is not symmetric. y[2,3] = 0.5, but y[3,2] = 0";
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y),
                   std::domain_error,
                   expected_msg);
}

TEST_F(ErrorHandlingMatrix, checkPosDefinite_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::math::check_pos_definite;

  y.resize(1,1);
  y << nan;
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                   std::domain_error,
                   "function: y is not positive definite: nan");
  
  y.resize(3,3);
  y << 2, -1, 0,
    -1, 2, -1,
    0, -1, 2;
  EXPECT_TRUE(check_pos_definite(function, 
                                 "y", y));
  for (int i = 0; i < y.rows(); i++)
    for (int j = 0; j < y.cols(); j++) {
      y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
      y(i,j) = nan;
      if (i >= j) {
        std::stringstream expected_msg;
        if (i == j)
          expected_msg << "function: y["
                       << j*y.cols() + i + 1 
                       << "] is nan, but must not be nan!";
        else
          expected_msg << "function: y is not symmetric. " 
                       << "y[" << j+1 << "," << i+1 << "] = " << y(j,i)
                       << ", but y[" << i+1 << "," << j+1 << "] = " << y(i,j);
        EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                         std::domain_error,
                         expected_msg.str());
      }
    }
  
  std::stringstream expected_msg;
  y << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  expected_msg << "function: y is not positive definite:\n"
               << y;
  EXPECT_THROW_MSG(check_pos_definite(function, "y", y), 
                   std::domain_error,
                   expected_msg.str());
}

