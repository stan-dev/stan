#include <stan/math/error_handling/matrix/check_std_vector_index.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkStdVectorIndexMatrix) {
  std::vector<double> y;
  y.push_back(5);
  y.push_back(5);
  y.push_back(5);
  y.push_back(5);
  double result;
  size_t i;
  
  i=2;
  y.resize(3);
  EXPECT_TRUE(stan::math::check_std_vector_index("checkStdVectorIndexMatrix(%1%)",
                                          i,y, "i", &result));
  i=3;
  EXPECT_TRUE(stan::math::check_std_vector_index("checkStdVectorIndexMatrix(%1%)",
                                          i,y, "i", &result));

  y.resize(2);
  EXPECT_THROW(stan::math::check_std_vector_index("checkStdVectorIndexMatrix(%1%)",i, y, "i", 
                                           &result), 
               std::domain_error);

  i=0;
  EXPECT_THROW(stan::math::check_std_vector_index("checkStdVectorIndexMatrix(%1%)",i, y, "i", 
                                              &result), 
               std::domain_error);
}
