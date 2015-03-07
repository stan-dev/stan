#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/err/check_consistent_size.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingScalar, checkConsistentSize_EigenVector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize";
  const char* name1 = "name1";
  

  Matrix<double,Dynamic,1> x(4);
  EXPECT_EQ(4U, size_of(x));
  EXPECT_TRUE(check_consistent_size(function, name1, x, 4U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 2U), 
                   std::invalid_argument,
                   "name1 has dimension = 4, expecting dimension = 2");

  x.resize(1);
  EXPECT_TRUE(check_consistent_size(function, name1, x, 1U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 2U), 
                   std::invalid_argument,
                   "name1 has dimension = 1, expecting dimension = 2");

  x.resize(0);
  EXPECT_TRUE(check_consistent_size(function, name1, x, 0U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 1U), 
                   std::invalid_argument,
                   "name1 has dimension = 0, expecting dimension = 1");
}


TEST(ErrorHandlingScalar, checkConsistentSize_StdVector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize";
  const char* name1 = "name1";
  

  std::vector<double> x(4);
  EXPECT_EQ(4U, size_of(x));
  EXPECT_TRUE(check_consistent_size(function, name1, x, 4U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 2U), 
                   std::invalid_argument,
                   "name1 has dimension = 4, expecting dimension = 2");

  x.resize(1);
  EXPECT_TRUE(check_consistent_size(function, name1, x, 1U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 2U), 
                   std::invalid_argument,
                   "name1 has dimension = 1, expecting dimension = 2");

  x.resize(0);
  EXPECT_TRUE(check_consistent_size(function, name1, x, 0U));
  EXPECT_THROW_MSG(check_consistent_size(function, name1, x, 1U), 
                   std::invalid_argument,
                   "name1 has dimension = 0, expecting dimension = 1");
}


TEST(ErrorHandlingScalar, checkConsistentSize_scalar) {
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize";
  const char* name1 = "name1";
  
  double x = 0;
  
  EXPECT_TRUE(check_consistent_size(function, name1, x, 4U));
}

TEST(ErrorHandlingScalar, checkConsistentSize_nan) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize";
  const char* name1 = "name1";
  
  double nan = std::numeric_limits<double>::quiet_NaN();

  Matrix<double,Dynamic,1> v1(4);
  v1 << nan,nan,4,nan;
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(function, name1, v1, 4U));
  EXPECT_THROW(check_consistent_size(function, name1, v1, 2U), 
               std::invalid_argument);
}
