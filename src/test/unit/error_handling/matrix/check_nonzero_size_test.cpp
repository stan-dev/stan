#include <stan/error_handling/matrix/check_nonzero_size.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingMatrix, checkNonzeroSizeMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  using stan::error_handling::check_nonzero_size;
  
  y.resize(3,3);
  EXPECT_TRUE(check_nonzero_size("checkNonzeroSize",
                                 "y", y));
  y.resize(2, 3);
  EXPECT_TRUE(check_nonzero_size("checkNonzeroSize",
                                 "y", y));

  y.resize(0,0);
  EXPECT_THROW_MSG(check_nonzero_size("checkNonzeroSize",
                                      "y", y),
                   std::domain_error, 
                   "has size 0");

  std::vector<double> a;
  a.push_back(3.0);
  a.push_back(3.0);
  a.push_back(3.0);
  a.push_back(3.0);


  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "a", a));

  a.resize(2);
  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "a", a));

  a.resize(0);
  EXPECT_THROW_MSG(stan::error_handling::check_nonzero_size("checkNonzeroSize", "a", a), 
                   std::domain_error,
                   "has size 0");
}

TEST(ErrorHandlingMatrix, checkNonzeroSizeMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "y", y));
  y.resize(2, 3);
  y << nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "y", y));

  y.resize(0,0);
  EXPECT_THROW_MSG(stan::error_handling::check_nonzero_size("checkNonzeroSize", "y", y), 
                   std::domain_error,
                   "has size 0");

  std::vector<double> a;
  a.push_back(nan);
  a.push_back(nan);
  a.push_back(nan);
  a.push_back(nan);

  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "a", a));

  a.resize(2);
  EXPECT_TRUE(stan::error_handling::check_nonzero_size("checkNonzeroSize",
                                                       "a", a));

  a.resize(0);
  EXPECT_THROW_MSG(stan::error_handling::check_nonzero_size("checkNonzeroSize", "a", a), 
                   std::domain_error,
                   "has size 0");
}
