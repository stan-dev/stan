#include <stan/error_handling/matrix/check_column_index.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkColumnIndexMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  size_t i;
  
  i=2;
  y.resize(3,3);
  EXPECT_TRUE(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                       "i", y, i));
  i=3;
  EXPECT_TRUE(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                       "i", y, i));

  y.resize(3, 2);
  EXPECT_THROW(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                        "i", y, i), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                        "i", y, i), 
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkColumnIndexMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();
  size_t i;
  
  i=2;
  y.resize(3,3);
  y << nan, nan, nan, nan, nan, nan, nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                       "i", y, i));
  i=3;
  EXPECT_TRUE(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                       "i", y, i));

  y.resize(3, 2);
  y << nan, nan, nan, nan, nan, nan;
  EXPECT_THROW(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                        "i", y, i), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::error_handling::check_column_index("checkColumnIndexMatrix",
                                                        "i", y, i), 
               std::domain_error);
}
