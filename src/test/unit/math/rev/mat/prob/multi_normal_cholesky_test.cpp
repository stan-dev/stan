#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_normal_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/multi_normal_cholesky_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/mat/fun/dot_self.hpp>
#include <stan/math/rev/mat/fun/columns_dot_self.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormalCholesky,MultiNormalVar) {
  using stan::math::var;
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<var,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::math::multi_normal_cholesky_log(y,mu,L).val());
}
