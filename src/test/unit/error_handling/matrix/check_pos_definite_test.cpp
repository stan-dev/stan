#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingMatrix, checkPosDefinite) {
  using stan::error_handling::check_pos_definite;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;

  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_pos_definite("checkPosDefinite", "y", y));

  y.resize(3,3);
  y << 
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;
  EXPECT_TRUE(check_pos_definite("checkPosDefinite", "y", y));
}

TEST(ErrorHandlingMatrix, checkPosDefinite_not_square) {
  using stan::error_handling::check_pos_definite;
  std::string expected_msg;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;

  y.resize(3, 4);
  expected_msg = "Rows of matrix (3) and columns of matrix (4) must match in size";
  EXPECT_THROW_MSG(check_pos_definite("checkPosDefinite", "y", y),
                   std::invalid_argument,
                   expected_msg);
}

TEST(ErrorHandlingMatrix, checkPosDefinite_0_size) {
  using stan::error_handling::check_pos_definite;
  std::string expected_msg;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;

  expected_msg = "y must have a positive size, but is 0; dimension size expression = rows";
  EXPECT_THROW_MSG(check_pos_definite("checkPosDefinite", "y", y),
                   std::invalid_argument,
                   expected_msg);
}

TEST(ErrorHandlingMatrix, checkPosDefinite_non_symmetric) {
  using stan::error_handling::check_pos_definite;
  std::string expected_msg;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(3,3);
  y <<
    1, 0, 0,
    0, 1, 0.5,
    0, 0, 1;
  
  expected_msg = "y is not symmetric. y[2,3] = 0.5, but y[3,2] = 0";
  EXPECT_THROW_MSG(check_pos_definite("checkPosDefinite", "y", y),
                   std::domain_error,
                   expected_msg);
}

TEST(ErrorHandlingMatrix, checkPosDefinite_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::error_handling::check_pos_definite;

  y.resize(1,1);
  y << nan;
  EXPECT_THROW(check_pos_definite("checkPosDefinite", 
                                  "y", y), 
               std::domain_error);
  
  y.resize(3,3);
  y << 2, -1, 0,
    -1, 2, -1,
    0, -1, 2;
  EXPECT_TRUE(check_pos_definite("checkPosDefinite", 
                                 "y", y));
  for (int i = 0; i < y.rows(); i++)
    for (int j = 0; j < y.cols(); j++) {
      y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
      y(i,j) = nan;
      if (i >= j)
        EXPECT_THROW(check_pos_definite("checkPosDefinite", 
                                        "y", y), 
                     std::domain_error);
    }

  y << 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0;
  EXPECT_THROW(check_pos_definite("checkPosDefinite", 
                                  "y", y), 
               std::domain_error);
}

