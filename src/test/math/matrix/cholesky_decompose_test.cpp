#include <stan/math/matrix/cholesky_decompose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, cholesky_decompose) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::cholesky_decompose;

  EXPECT_NO_THROW(cholesky_decompose(m0));
  EXPECT_THROW(cholesky_decompose(m1),std::domain_error);
}

TEST(MathMatrix, cholesky_decompose_exception) {
  stan::math::matrix_d m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}
