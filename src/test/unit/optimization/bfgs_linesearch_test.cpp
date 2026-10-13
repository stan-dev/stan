#include <gtest/gtest.h>
#include <stan/optimization/bfgs_linesearch.hpp>
#include <stan/math/prim.hpp>
#include <cmath>
#include <stdexcept>

TEST(OptimizationBfgsLinesearch, CubicInterp) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double xMin;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0 * x0 * x0 / 3.0 - x0;
    const double df0 = x0 * x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1 * x1 * x1 / 3.0 - x1;
      const double df1 = x1 * x1 - 1.0;

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -3.0, 2.0);
      EXPECT_NEAR(-3.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -3.0, 0.0);
      EXPECT_NEAR(-3.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, -1.0, 2.0);
      EXPECT_NEAR(1.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, 0.0, 2.0);
      EXPECT_NEAR(1.0, xMin, 1e-8);

      xMin = CubicInterp(x0, f0, df0, x1, f1, df1, 0.5, 1.5);
      EXPECT_NEAR(1.0, xMin, 1e-8);
    }
  }
}

TEST(OptimizationBfgsLinesearch, CubicInterp_6arg) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  double x;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0 * x0 * x0 / 3.0 - x0;
    const double df0 = x0 * x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1 * x1 * x1 / 3.0 - x1;
      const double df1 = x1 * x1 - 1.0;

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -3.0 - x0, 2.0 - x0);
      EXPECT_NEAR(-3.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -3.0 - x0, 0.0 - x0);
      EXPECT_NEAR(-3.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, -1.0 - x0, 2.0 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, 0.0 - x0, 2.0 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);

      x = CubicInterp(df0, x1 - x0, f1 - f0, df1, 0.5 - x0, 1.5 - x0);
      EXPECT_NEAR(1.0, x0 + x, 1e-8);
    }
  }
}

class linesearch_testfunc {
 public:
  double operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x) {
    return x.dot(x) - 1.0;
  }
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    f = x.dot(x) - 1.0;
    g = 2.0 * x;
    return 0;
  }
};

TEST(OptimizationBfgsLinesearch, WolfLSZoom) {
  using stan::optimization::WolfLSZoom;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;

  linesearch_testfunc func1;
  Eigen::Matrix<double, -1, 1> x0, x1;
  double f0, f1;
  Eigen::Matrix<double, -1, 1> p, gradx0, gradx1;
  double alpha;
  int ret;

  x0.setOnes(5, 1);
  p = -gradx0;

  func1(x0, f0, gradx0);

  p = -gradx0;

  double dfp = gradx0.dot(p);
  alpha = 2.0;
  x1 = x0 + alpha * p;
  func1(x1, f1, gradx1);

  double dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1, func1, x0, f0, dfp, c1 * dfp,
                   c2 * dfp, p, minAlpha, f0, dfp, alpha, f1, dfp2, 1e-16);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  x1 = x0 + alpha * p;
  func1(x1, f1, gradx1);

  dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1, func1, x0, f0, dfp, c1 * dfp,
                   c2 * dfp, p, minAlpha, f0, dfp, alpha, f1, dfp2, 1e-16);

  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

TEST(OptimizationBfgsLinesearch, wolfeLineSearch) {
  using stan::optimization::WolfeLineSearch;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;
  static const double maxLSIts = 20;
  static const double maxLSRestarts = 10;

  linesearch_testfunc func1;
  Eigen::Matrix<double, -1, 1> x0, x1;
  double f0, f1;
  Eigen::Matrix<double, -1, 1> p, gradx0, gradx1;
  double alpha;
  int ret;

  x0.setOnes(5, 1);
  func1(x0, f0, gradx0);

  p = -gradx0;

  alpha = 2.0;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.5, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));

  alpha = 0.25;
  ret = WolfeLineSearch(func1, alpha, x1, f1, gradx1, p, x0, f0, gradx0, c1, c2,
                        minAlpha, maxLSIts, maxLSRestarts);
  EXPECT_EQ(0, ret);
  EXPECT_NEAR(0.25, alpha, 1e-8);
  EXPECT_NEAR(0, (x1 - (x0 + alpha * p)).norm(), 1e-8);
  EXPECT_EQ(f1, func1(x1));
  EXPECT_LE(f1, f0 + c1 * alpha * p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)), c2 * std::fabs(p.dot(gradx0)));
}

// A 1-D test function for #3229: slope -1 below jump_at, value and slope
// 1e300 from jump_at on, and failed evaluations for fail_lo <= x < fail_hi.
// A line search towards jump_at can bracket the jump, but no point meets
// the Wolfe conditions. The evaluation count is limited, so that an endless
// loop makes a test fail instead of hang.
class linesearch_jumpfunc {
 public:
  linesearch_jumpfunc(double jump_at, double fail_lo, double fail_hi)
      : jump_at_(jump_at), fail_lo_(fail_lo), fail_hi_(fail_hi) {}
  int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x, double &f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
    if (++n_evals_ > 100000)
      throw std::runtime_error("line search does not terminate");
    g.resize(x.size());
    if (x[0] >= fail_lo_ && x[0] < fail_hi_)
      return 1;
    f = x[0] < jump_at_ ? -x[0] : 1e300;
    g[0] = x[0] < jump_at_ ? -1.0 : 1e300;
    return 0;
  }
  int n_evals_ = 0;

 private:
  double jump_at_, fail_lo_, fail_hi_;
};

TEST(OptimizationBfgsLinesearch, wolfeLineSearch_jump_terminates) {
  // the zoom bisects towards the jump at 10 until alo and ahi are adjacent
  // doubles; it then looped forever (#3229)
  using stan::optimization::WolfeLineSearch;
  linesearch_jumpfunc func(10.0, 0.0, 0.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx0, gradx1;
  double f0, f1;
  x0 << 0.0;
  func(x0, f0, gradx0);
  p << 1.0;
  double alpha = 1.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfeLineSearch(func, alpha, x1, f1, gradx1, p, x0, f0,
                                        gradx0, 1e-4, 0.9, 1e-16, 20.0, 10.0));
  EXPECT_EQ(1, ret);
  EXPECT_LT(func.n_evals_, 1000);
}

TEST(OptimizationBfgsLinesearch, WolfLSZoom_adjacent_bracket) {
  // alo and ahi are adjacent doubles and every trial step rounds to ahi;
  // this looped forever (#3229)
  using stan::optimization::WolfLSZoom;
  linesearch_jumpfunc func(10.0, 0.0, 0.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx1;
  x0 << 0.0;
  p << 1.0;
  const double f0 = 0.0, dfp = -1.0;
  const double alo = std::nextafter(10.0, 0.0), ahi = 10.0;
  double alpha = ahi, f1 = 0.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfLSZoom(alpha, x1, f1, gradx1, func, x0, f0, dfp,
                                   1e-4 * dfp, 0.9 * dfp, p, alo, -alo, -1.0,
                                   ahi, 1e300, 1e300, 1e-16));
  EXPECT_EQ(1, ret);
  EXPECT_LT(func.n_evals_, 1000);
}

TEST(OptimizationBfgsLinesearch, WolfLSZoom_failing_evaluations) {
  // evaluations fail for 10 <= x < 12: the trial steps are halved towards
  // alo = 10 - 1 ulp, reach 10 and then stay there; this looped forever
  // (#3229)
  using stan::optimization::WolfLSZoom;
  linesearch_jumpfunc func(10.0, 10.0, 12.0);
  Eigen::Matrix<double, -1, 1> x0(1), x1, p(1), gradx1;
  x0 << 0.0;
  p << 1.0;
  const double f0 = 0.0, dfp = -1.0;
  const double alo = std::nextafter(10.0, 0.0), ahi = 13.0;
  double alpha = ahi, f1 = 0.0;
  int ret = 0;
  EXPECT_NO_THROW(ret = WolfLSZoom(alpha, x1, f1, gradx1, func, x0, f0, dfp,
                                   1e-4 * dfp, 0.9 * dfp, p, alo, -alo, -1.0,
                                   ahi, 1e300, 1e300, 1e-16));
  EXPECT_EQ(1, ret);
  EXPECT_LT(func.n_evals_, 1000);
}
