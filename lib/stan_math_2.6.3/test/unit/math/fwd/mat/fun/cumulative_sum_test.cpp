#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/cumulative_sum.hpp>
#include <stan/math/fwd/core.hpp>

template <typename T>
void test_cumulative_sum() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  c[0].d_ = 1.0;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0].val_,d[0].val_);
  EXPECT_FLOAT_EQ(1.0, d[0].d_);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  e[0].d_ = 2.0;  e[1].d_ = 1.0;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0].val_,f[0].val_);
  EXPECT_FLOAT_EQ((e[0] + e[1]).val_, f[1].val_);
  EXPECT_FLOAT_EQ(2.0, f[0].d_);
  EXPECT_FLOAT_EQ(3.0, f[1].d_);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  g[0].d_ = 4.0;  g[1].d_ = 2.0;   g[2].d_ = 3.0;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0].val_,h[0].val_);
  EXPECT_FLOAT_EQ((g[0] + g[1]).val_, h[1].val_);
  EXPECT_FLOAT_EQ((g[0] + g[1] + g[2]).val_, h[2].val_);
  EXPECT_FLOAT_EQ(4.0, h[0].d_);
  EXPECT_FLOAT_EQ(6.0, h[1].d_);
  EXPECT_FLOAT_EQ(9.0, h[2].d_);
}
template <typename T>
void test_cumulative_sum3() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  c[0].d_ = 1.0;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0].val_.val(),d[0].val_.val());
  EXPECT_FLOAT_EQ(1.0, d[0].d_.val());

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  e[0].d_ = 2.0;  e[1].d_ = 1.0;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0].val_.val(),f[0].val_.val());
  EXPECT_FLOAT_EQ((e[0] + e[1]).val_.val(), f[1].val_.val());
  EXPECT_FLOAT_EQ(2.0, f[0].d_.val());
  EXPECT_FLOAT_EQ(3.0, f[1].d_.val());

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  g[0].d_ = 4.0;  g[1].d_ = 2.0;   g[2].d_ = 3.0;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0].val_.val(),h[0].val_.val());
  EXPECT_FLOAT_EQ((g[0] + g[1]).val_.val(), h[1].val_.val());
  EXPECT_FLOAT_EQ((g[0] + g[1] + g[2]).val_.val(), h[2].val_.val());
  EXPECT_FLOAT_EQ(4.0, h[0].d_.val());
  EXPECT_FLOAT_EQ(6.0, h[1].d_.val());
  EXPECT_FLOAT_EQ(9.0, h[2].d_.val());
}
TEST(AgradFwdMatrixCumulativeSum, fd) {
  using stan::math::fvar;
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<fvar<double> >(0)).size());

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<fvar<double>,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum<std::vector<fvar<double> > >();
  test_cumulative_sum<Eigen::Matrix<fvar<double>,Eigen::Dynamic,1> >();
  test_cumulative_sum<Eigen::Matrix<fvar<double>,1,Eigen::Dynamic> >();
}
TEST(AgradFwdMatrixCumulativeSum, ffd) {
  using stan::math::fvar;
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<fvar<fvar<double> > >(0)).size());

  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<fvar<fvar<double> >,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum3<std::vector<fvar<fvar<double> > > >();
  test_cumulative_sum3<Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1> >();
  test_cumulative_sum3<Eigen::Matrix<fvar<fvar<double> >,1,Eigen::Dynamic> >();
}

