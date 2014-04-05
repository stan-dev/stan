#include <stan/optimization/bfgs.hpp>
#include <gtest/gtest.h>

TEST(OptimizationBFGS, cubic_interp) {
  using stan::optimization::CubicInterp;
  const unsigned int nVals = 5;
  const double xVals[5] = { -2.0, -1.0, 0.0, 1.0, 2.0 };
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


