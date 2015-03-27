#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(MathMatrix, cholesky_decompose) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::cholesky_decompose;

  EXPECT_NO_THROW(cholesky_decompose(m0));
  EXPECT_THROW_MSG(cholesky_decompose(m1),
                   std::invalid_argument,
                   "Expecting a square matrix");
}

TEST(MathMatrix, cholesky_decompose_exception) {
  stan::math::matrix_d m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_THROW_MSG(stan::math::cholesky_decompose(m),
                   std::domain_error,
                   "Cholesky decomposition of m failed");

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW_MSG(stan::math::cholesky_decompose(m),
                   std::invalid_argument,
                   "Expecting a square matrix");

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW_MSG(stan::math::cholesky_decompose(m),
                   std::domain_error,
                   "is not symmetric");
}
