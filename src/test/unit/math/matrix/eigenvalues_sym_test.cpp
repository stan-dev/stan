#include <stan/math/matrix/eigenvalues_sym.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, eigenvalues_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::domain_error);
  EXPECT_THROW(eigenvalues_sym(m1),std::domain_error);
}

