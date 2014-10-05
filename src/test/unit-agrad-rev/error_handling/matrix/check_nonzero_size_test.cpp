#include <stan/agrad/rev/var.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(AgradRevErrorHandlingMatrix, checkNonzeroSizeMatrix) {
  using stan::agrad::var;
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> y;
  using stan::math::check_nonzero_size;
  var result;
  
  y.resize(3,3);
  EXPECT_TRUE(check_nonzero_size("checkNonzeroSize(%1%)",
                                 y, "y", &result));
  y.resize(2, 3);
  EXPECT_TRUE(check_nonzero_size("checkNonzeroSize(%1%)",
                                             y, "y", &result));

  y.resize(0,0);
  EXPECT_THROW_MSG(check_nonzero_size("checkNonzeroSize(%1%)",
                                      y, "y", &result),
                   std::domain_error, "has size 0");

  std::vector<var> a;
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
  EXPECT_THROW_MSG(stan::math::check_nonzero_size("checkNonzeroSize(%1%)", a, "a", 
                                                  &result), 
                   std::domain_error,
                   "has size 0");
}

TEST(AgradRevErrorHandlingMatrix, checkNonzeroSizeMatrix_nan) {
  using stan::agrad::var;
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> y;
  var result;
  var nan = std::numeric_limits<var>::quiet_NaN();

  y.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             y, "y", &result));
  y.resize(2, 3);
  y << nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             y, "y", &result));

  y.resize(0,0);
  EXPECT_THROW_MSG(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",y, "y", 
                                                  &result), 
                   std::domain_error,
                   "has size 0");

  std::vector<var> a;
  a.push_back(nan);
  a.push_back(nan);
  a.push_back(nan);
  a.push_back(nan);

  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             a, "a", &result));

  a.resize(2);
  EXPECT_TRUE(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",
                                             a, "a", &result));

  a.resize(0);
  EXPECT_THROW_MSG(stan::math::check_nonzero_size("checkNonzeroSize(%1%)",a, "a", 
                                                  &result), 
                   std::domain_error,
                   "has size 0");
}
