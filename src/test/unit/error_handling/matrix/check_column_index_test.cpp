#include <stan/error_handling/matrix/check_column_index.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkColumnIndexMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  size_t i;
  
  i=2;
  y.resize(3,3);
  EXPECT_TRUE(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",
                                          i,y, "i", &result));
  i=3;
  EXPECT_TRUE(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",
                                          i,y, "i", &result));

  y.resize(3, 2);
  EXPECT_THROW(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",i, y, "i", 
                                           &result), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",i, y, "i", 
                                              &result), 
               std::domain_error);
}

TEST(MathErrorHandlingMatrix, checkColumnIndexMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double result;
  size_t i;
  
  i=2;
  y.resize(3,3);
  y << nan, nan, nan, nan, nan, nan, nan, nan, nan;
  EXPECT_TRUE(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",
                                          i,y, "i", &result));
  i=3;
  EXPECT_TRUE(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",
                                          i,y, "i", &result));

  y.resize(3, 2);
  y << nan, nan, nan, nan, nan, nan;
  EXPECT_THROW(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",i, y, "i", 
                                           &result), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::math::check_column_index("checkColumnIndexMatrix(%1%)",i, y, "i", 
                                              &result), 
               std::domain_error);
}
