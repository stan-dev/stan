#include <stan/math/prim/mat/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, eigenvalues_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::invalid_argument);
  EXPECT_THROW(eigenvalues_sym(m1),std::invalid_argument);
}

