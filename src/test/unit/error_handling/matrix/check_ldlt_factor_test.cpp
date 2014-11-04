#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, CheckLDLTFactor_nan) {
  using stan::error_handling::check_ldlt_factor;

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x(2,2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_x;
  double result;

  x << nan, 1, 1, 3;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix(%1%)", 
                                 ldlt_x, "ldlt_x", &result), 
               std::domain_error);

  x << 3, nan, 1, 3;
  ldlt_x.compute(x);
  ASSERT_TRUE(ldlt_x.success());
  EXPECT_NO_THROW(check_ldlt_factor("checkLDLTFactorMatrix(%1%)", 
                                    ldlt_x, "ldlt_x", &result));

  x << 3, 1, nan, 3;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix(%1%)", 
                                 ldlt_x, "ldlt_x", &result), 
               std::domain_error);

  x << 3, 1, 1, nan;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix(%1%)", 
                                 ldlt_x, "ldlt_x", &result), 
               std::domain_error);
}
