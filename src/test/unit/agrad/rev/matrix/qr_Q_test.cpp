#include <stan/math/matrix/qr_Q.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

TEST(MathMatrix, qr_Q) {
  stan::agrad::matrix_v m0(0,0);
  stan::agrad::matrix_v m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::qr_Q;
  using stan::math::transpose;
  EXPECT_THROW(qr_Q(m0),std::domain_error);
  EXPECT_NO_THROW(qr_Q(m1));
  EXPECT_THROW(qr_Q(transpose(m1)),std::domain_error);
}
