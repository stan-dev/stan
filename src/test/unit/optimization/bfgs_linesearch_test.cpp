#include <gtest/gtest.h>
#include <stan/optimization/bfgs_linesearch.hpp>
#include <stan/math/prim/mat.hpp>

TEST(OptimizationBfgsLinesearch, CubicInterp) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = { -2.0, -1.0, 0.0, 1.0, 2.0 };
  double xMin;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0*x0*x0/3.0 - x0;
    const double df0 = x0*x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1*x1*x1/3.0 - x1;
      const double df1 = x1*x1 - 1.0;

      xMin = CubicInterp(x0,f0,df0,x1,f1,df1,-3.0,2.0);
      EXPECT_NEAR(-3.0,xMin,1e-8);

      xMin = CubicInterp(x0,f0,df0,x1,f1,df1,-3.0,0.0);
      EXPECT_NEAR(-3.0,xMin,1e-8);

      xMin = CubicInterp(x0,f0,df0,x1,f1,df1,-1.0,2.0);
      EXPECT_NEAR(1.0,xMin,1e-8);

      xMin = CubicInterp(x0,f0,df0,x1,f1,df1,0.0,2.0);
      EXPECT_NEAR(1.0,xMin,1e-8);

      xMin = CubicInterp(x0,f0,df0,x1,f1,df1,0.5,1.5);
      EXPECT_NEAR(1.0,xMin,1e-8);
    }
  }
}

TEST(OptimizationBfgsLinesearch, CubicInterp_6arg) {
  using stan::optimization::CubicInterp;
  static const unsigned int nVals = 5;
  static const double xVals[5] = { -2.0, -1.0, 0.0, 1.0, 2.0 };
  double x;

  for (unsigned int i = 0; i < nVals; i++) {
    const double &x0 = xVals[i];
    const double f0 = x0*x0*x0/3.0 - x0;
    const double df0 = x0*x0 - 1.0;
    for (unsigned int j = 0; j < nVals; j++) {
      if (i == j)
        continue;

      const double &x1 = xVals[j];
      const double f1 = x1*x1*x1/3.0 - x1;
      const double df1 = x1*x1 - 1.0;

      x = CubicInterp(df0,x1-x0,f1-f0,df1,-3.0-x0,2.0-x0);
      EXPECT_NEAR(-3.0,x0+x,1e-8);

      x = CubicInterp(df0,x1-x0,f1-f0,df1,-3.0-x0,0.0-x0);
      EXPECT_NEAR(-3.0,x0+x,1e-8);

      x = CubicInterp(df0,x1-x0,f1-f0,df1,-1.0-x0,2.0-x0);
      EXPECT_NEAR(1.0,x0+x,1e-8);

      x = CubicInterp(df0,x1-x0,f1-f0,df1,0.0-x0,2.0-x0);
      EXPECT_NEAR(1.0,x0+x,1e-8);

      x = CubicInterp(df0,x1-x0,f1-f0,df1,0.5-x0,1.5-x0);
      EXPECT_NEAR(1.0,x0+x,1e-8);
    }
  }
}

class linesearch_testfunc {
public:
  double operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x) {
    return x.dot(x) - 1.0;
  }
  int operator()(const Eigen::Matrix<double,Eigen::Dynamic,1> &x,
                 double &f, Eigen::Matrix<double,Eigen::Dynamic,1> &g) {
    f = x.dot(x) - 1.0;
    g = 2.0*x;
    return 0;
  }
};

TEST(OptimizationBfgsLinesearch, WolfLSZoom) {
  using stan::optimization::WolfLSZoom;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;

  linesearch_testfunc func1;
  Eigen::Matrix<double,-1,1> x0,x1;
  double f0,f1;
  Eigen::Matrix<double,-1,1> p, gradx0,gradx1;
  double alpha;
  int ret;

  x0.setOnes(5,1);
  p = -gradx0;

  func1(x0,f0,gradx0);

  p = -gradx0;

  double dfp = gradx0.dot(p);
  alpha = 2.0;
  x1 = x0 + alpha*p;
  func1(x1,f1,gradx1);

  double dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1,
                   func1, x0, f0, dfp,
                   c1*dfp, c2*dfp, p,
                   minAlpha, f0, dfp,
                   alpha, f1, dfp2, 1e-16);
  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.5,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  x1 = x0 + alpha*p;
  func1(x1,f1,gradx1);

  dfp2 = gradx1.dot(p);

  ret = WolfLSZoom(alpha, x1, f1, gradx1,
                   func1, x0, f0, dfp,
                   c1*dfp, c2*dfp, p,
                   minAlpha, f0, dfp,
                   alpha, f1, dfp2, 1e-16);

  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.5,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));
}


TEST(OptimizationBfgsLinesearch, wolfeLineSearch) {
  using stan::optimization::WolfeLineSearch;

  static const double c1 = 1e-4;
  static const double c2 = 0.9;
  static const double minAlpha = 1e-16;
  static const double maxLSIts = 20;
  static const double maxLSRestarts = 10;

  linesearch_testfunc func1;
  Eigen::Matrix<double,-1,1> x0,x1;
  double f0,f1;
  Eigen::Matrix<double,-1,1> p, gradx0,gradx1;
  double alpha;
  int ret;

  x0.setOnes(5,1);
  func1(x0,f0,gradx0);

  p = -gradx0;

  alpha = 2.0;
  ret = WolfeLineSearch(func1, alpha,
                        x1, f1, gradx1,
                        p, x0, f0, gradx0,
                        c1, c2, minAlpha,
                        maxLSIts, maxLSRestarts);
  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.5,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));

  alpha = 10.0;
  ret = WolfeLineSearch(func1, alpha,
                        x1, f1, gradx1,
                        p, x0, f0, gradx0,
                        c1, c2, minAlpha,
                        maxLSIts, maxLSRestarts);
  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.5,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));

  alpha = 0.25;
  ret = WolfeLineSearch(func1, alpha,
                        x1, f1, gradx1,
                        p, x0, f0, gradx0,
                        c1, c2, minAlpha,
                        maxLSIts, maxLSRestarts);
  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.25,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));
}
