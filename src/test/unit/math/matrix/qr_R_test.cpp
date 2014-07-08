#include <stan/math/matrix/qr_Q.hpp>
#include <stan/math/matrix/qr_R.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, qr_R) {
  stan::math::matrix_d m0(0,0);
  stan::math::matrix_d m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::qr_R;
  using stan::math::qr_Q;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));

  stan::math::matrix_d m2(3,2);
  m2 = qr_Q(m1) * qr_R(m1);
  for (unsigned int i=0; i<m1.rows(); i++) {
    for (unsigned int j=0; j<m1.cols(); j++) {
      EXPECT_NEAR(m1(i,j), m2(i,j), 1e-8);
    }
  }
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);
}
