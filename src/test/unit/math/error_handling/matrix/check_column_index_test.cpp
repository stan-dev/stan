#include <stan/math/error_handling/matrix/check_column_index.hpp>
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
