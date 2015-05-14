#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/cumulative_sum.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>
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
void test_cumulative_sum2() {
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
  VEC grad = cgrad(f[0].val(),e[0].val(),e[1].val());
  EXPECT_EQ(2U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);

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
  grad = cgrad(h[2].val(),g[0].val(),g[1].val(),g[2].val());
  EXPECT_EQ(3U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(1.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
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
template <typename T>
void test_cumulative_sum4() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  c[0].d_ = 1.0;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0].val_.val().val(),d[0].val_.val().val());
  EXPECT_FLOAT_EQ(1.0, d[0].d_.val().val());

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  e[0].d_ = 2.0;  e[1].d_ = 1.0;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0].val_.val().val(),f[0].val_.val().val());
  EXPECT_FLOAT_EQ((e[0] + e[1]).val_.val().val(), f[1].val_.val().val());
  EXPECT_FLOAT_EQ(2.0, f[0].d_.val().val());
  EXPECT_FLOAT_EQ(3.0, f[1].d_.val().val());
  VEC grad = cgrad(f[0].val().val(),e[0].val().val(),e[1].val().val());
  EXPECT_EQ(2U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  g[0].d_ = 4.0;  g[1].d_ = 2.0;   g[2].d_ = 3.0;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0].val_.val().val(),h[0].val_.val().val());
  EXPECT_FLOAT_EQ((g[0] + g[1]).val_.val().val(), h[1].val_.val().val());
  EXPECT_FLOAT_EQ((g[0] + g[1] + g[2]).val_.val().val(), h[2].val_.val().val());
  EXPECT_FLOAT_EQ(4.0, h[0].d_.val().val());
  EXPECT_FLOAT_EQ(6.0, h[1].d_.val().val());
  EXPECT_FLOAT_EQ(9.0, h[2].d_.val().val());
  grad = cgrad(h[2].val().val(),g[0].val().val(),g[1].val().val(),g[2].val().val());
  EXPECT_EQ(3U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(1.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
}
TEST(AgradMixMatrixCumulativeSum, fv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<fvar<var> >(0)).size());

  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<fvar<var>,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum2<std::vector<fvar<var> > >();
  test_cumulative_sum2<Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> >();
  test_cumulative_sum2<Eigen::Matrix<fvar<var>,1,Eigen::Dynamic> >();
}
TEST(AgradMixMatrixCumulativeSum, ffv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<fvar<fvar<var> > >(0)).size());

  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<fvar<fvar<var> >,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum4<std::vector<fvar<fvar<var> > > >();
  test_cumulative_sum4<Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> >();
  test_cumulative_sum4<Eigen::Matrix<fvar<fvar<var> >,1,Eigen::Dynamic> >();
}
