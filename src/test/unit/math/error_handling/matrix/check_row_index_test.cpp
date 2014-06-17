#include <stan/math/error_handling/matrix/check_row_index.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkRowIndexMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  size_t i;
  
  i=2;
  y.resize(3,3);
  EXPECT_TRUE(stan::math::check_row_index("checkRowIndexMatrix(%1%)",
                                          i,y, "i", &result));
  i=3;
  EXPECT_TRUE(stan::math::check_row_index("checkRowIndexMatrix(%1%)",
                                          i,y, "i", &result));

  y.resize(2, 3);
  EXPECT_THROW(stan::math::check_row_index("checkRowIndexMatrix(%1%)",i, y, "i", 
                                           &result), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::math::check_row_index("checkRowIndexMatrix(%1%)",i, y, "i", 
                                           &result), 
               std::domain_error);
}
