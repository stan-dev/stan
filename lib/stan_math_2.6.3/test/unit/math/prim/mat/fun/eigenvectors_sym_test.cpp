#include <stan/math/prim/mat/fun/eigenvectors_sym.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, eigenvectors_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::invalid_argument);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::invalid_argument);
}

