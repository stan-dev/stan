#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkNonzeroSizeMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             y, "y", &result));
  y.resize(2, 3);
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             y, "y", &result));

  y.resize(0,0);
  EXPECT_THROW(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",y, "y", 
                                              &result), 
               std::domain_error);

  std::vector<double> a;
  a.push_back(3.0);
  a.push_back(3.0);
  a.push_back(3.0);
  a.push_back(3.0);


  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             a, "a", &result));

  a.resize(2);
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             a, "a", &result));

  a.resize(0);
  EXPECT_THROW(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",a, "a", 
                                              &result), 
               std::domain_error);
}
