#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, CheckLDLTFactor_nan) {
  using stan::math::check_ldlt_factor;

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x(2,2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_x;

  x << nan, 1, 1, 3;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix", 
                                 "ldlt_x", ldlt_x), 
               std::domain_error);

  x << 3, nan, 1, 3;
  ldlt_x.compute(x);
  ASSERT_TRUE(ldlt_x.success());
  EXPECT_NO_THROW(check_ldlt_factor("checkLDLTFactorMatrix", 
                                    "ldlt_x", ldlt_x));

  x << 3, 1, nan, 3;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix", 
                                 "ldlt_x", ldlt_x), 
               std::domain_error);

  x << 3, 1, 1, nan;
  ldlt_x.compute(x);
  ASSERT_FALSE(ldlt_x.success());
  EXPECT_THROW(check_ldlt_factor("checkLDLTFactorMatrix", 
                                 "ldlt_x", ldlt_x), 
               std::domain_error);
}
