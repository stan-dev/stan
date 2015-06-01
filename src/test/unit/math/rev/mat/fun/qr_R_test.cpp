#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/qr_R.hpp>
#include <stan/math/prim/mat/fun/qr_Q.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

TEST(MathMatrix, qr_R) {
  stan::math::matrix_v m0(0,0);
  stan::math::matrix_v m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::qr_R;
  using stan::math::qr_Q;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::invalid_argument);
  EXPECT_NO_THROW(qr_R(m1));

  stan::math::matrix_v m2(3,2);
  m2 = qr_Q(m1) * qr_R(m1);
  for (unsigned int i=0; i<m1.rows(); i++) {
    for (unsigned int j=0; j<m1.cols(); j++) {
      EXPECT_NEAR(m1(i,j).val(), m2(i,j).val(), 1e-8);
    }
  }
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);
}
