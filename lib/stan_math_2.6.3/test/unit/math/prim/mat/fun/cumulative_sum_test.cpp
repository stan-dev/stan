#include <stan/math/prim/mat/fun/cumulative_sum.hpp>
#include <gtest/gtest.h>

template <typename T>
void test_cumulative_sum() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0],d[0]);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0],f[0]);
  EXPECT_FLOAT_EQ(e[0] + e[1], f[1]);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0],h[0]);
  EXPECT_FLOAT_EQ(g[0] + g[1], h[1]);
  EXPECT_FLOAT_EQ(g[0] + g[1] + g[2], h[2]);
}

TEST(MathMatrix, cumulative_sum) {
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<double>(0)).size());

  Eigen::Matrix<double,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<double,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum<std::vector<double> >();
  test_cumulative_sum<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_cumulative_sum<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}
