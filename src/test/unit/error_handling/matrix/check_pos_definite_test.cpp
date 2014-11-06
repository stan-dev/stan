#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkPosDefiniteMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::error_handling::check_pos_definite;

  y.resize(1,1);
  y << nan;
  EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", 
                                              "y", y), 
               std::domain_error);
  
  y.resize(3,3);
  y << 2, -1, 0,
    -1, 2, -1,
    0, -1, 2;
  EXPECT_TRUE(check_pos_definite("checkPosDefiniteMatrix", 
                                              "y", y));
  for (int i = 0; i < y.rows(); i++)
    for (int j = 0; j < y.cols(); j++) {
      y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
      y(i,j) = nan;
      if (i >= j)
        EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", 
                                        "y", y), 
                     std::domain_error);
    }

  y << 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0;
  EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", 
                                              "y", y), 
               std::domain_error);
}

