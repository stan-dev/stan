#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <cmath>
#include <stan/math/rev/mat/fun/log_sum_exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/log.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using std::vector;

template <int R, int C>
void test_log_sum_exp_matrix(const Matrix<double,R,C>& m) {
  using std::exp;
  
  vector<var> x_expected(m.size());
  for (int i = 0; i < m.size(); ++i)
    x_expected[i] = m(i);
  var sum_exp(0);
  for (int i = 0; i < m.size(); ++i)
    sum_exp += exp(x_expected[i]);
  var f_expected = log(sum_exp);
  double val_expected = f_expected.val();
  vector<double> g_expected(m.size());
  f_expected.grad(x_expected,g_expected);
  
  
  vector<var> x(m.size());
  for (int i = 0; i < m.size(); ++i)
    x[i] = m(i);
  Matrix<var,R,C> mv(m.rows(),m.cols());
  for (int i = 0; i < m.size(); ++i)
    mv(i) = x[i];
  var f = log_sum_exp(mv);
  double val = f.val();
  vector<double> g(m.size());
  f.grad(x,g);

  EXPECT_FLOAT_EQ(val_expected, val);
  EXPECT_EQ(g_expected.size(), g.size());
  for (size_t i = 0; i < g.size(); ++i)
    EXPECT_FLOAT_EQ(g_expected[i], g[i]);
}

TEST(AgradRev,logSumExpMatrix) {
  Matrix<double,Dynamic,1> a(2);
  a << 5, 2;
  test_log_sum_exp_matrix(a);

  Matrix<double,Dynamic,1> b(1);
  b << 0;
  test_log_sum_exp_matrix(b);

  Matrix<double,1,Dynamic> c(3);
  c << 4.9, -12, 1.7;
  test_log_sum_exp_matrix(c);

  Matrix<double,Dynamic,Dynamic> d(3,2);
  d << -1, -2, -4, 5, 6, 4;
  test_log_sum_exp_matrix(d);
  

  
}
