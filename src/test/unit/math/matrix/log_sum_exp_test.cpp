#include <stan/math/matrix/log_sum_exp.hpp>
#include <gtest/gtest.h>

template <int R, int C>
void test_log_sum_exp(const Eigen::Matrix<double,R,C>& as) {
  using std::log;
  using std::exp;
  using stan::math::log_sum_exp;
  double sum_exp = 0.0;
  for (int n = 0; n < as.size(); ++n)
    sum_exp += exp(as(n));
  EXPECT_FLOAT_EQ(log(sum_exp),
                  log_sum_exp(as));
}

TEST(MathFunctions, log_sum_exp) {
  using stan::math::log_sum_exp;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,Dynamic> m(3,2);
  m << 1, 2, 3, 4, 5, 6;
  test_log_sum_exp(m);

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;
  test_log_sum_exp(v);

  Matrix<double,Dynamic,1> rv(3);
  rv << 1, 2, 3;
  test_log_sum_exp(rv);


  Matrix<double,Dynamic,Dynamic> m_trivial(1,1);
  m_trivial << 2;
  EXPECT_FLOAT_EQ(2, log_sum_exp(m_trivial));

}

