#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/prob/expect_eq_diffs.hpp>
#include <stan/math/prim/mat/prob/categorical_log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/prim/mat/err/check_simplex.hpp>

template <typename T_prob>
void expect_propto(unsigned int n1, T_prob theta1, 
                   unsigned int n2, T_prob theta2, 
                   std::string message) {
  expect_eq_diffs(stan::math::categorical_log<false>(n1, theta1),
                  stan::math::categorical_log<false>(n2, theta2),
                  stan::math::categorical_log<true>(n1, theta1),
                  stan::math::categorical_log<true>(n2, theta2),
                  message);
}

using stan::math::var;
using Eigen::Dynamic;
using Eigen::Matrix;


TEST(AgradDistributionsCategorical,Propto) {
  unsigned int n;
  Matrix<var,Dynamic,1> theta1(3,1);
  theta1 << 0.3, 0.5, 0.2;
  Matrix<var,Dynamic,1> theta2(3,1);
  theta2 << 0.1, 0.2, 0.7;
  
  n = 1;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");

  n = 2;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");

  n = 3;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");
}
