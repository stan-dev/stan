#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/bfgs_linesearch.hpp>

#include <test/test-models/no-main/optimization/rosenbrock.cpp>

#include <gtest/gtest.h>

TEST(OptimizationBFGS, cubic_interp) {
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

TEST(OptimizationBFGS, wolfe_line_search) {
  using stan::optimization::WolfeLineSearch;

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
  func1(x0,f0,gradx0);

  p = -gradx0;

  alpha = 2.0;
  ret = WolfeLineSearch(func1, alpha,
                        x1, f1, gradx1,
                        p, x0, f0, gradx0,
                        c1, c2, minAlpha);
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
                        c1, c2, minAlpha);
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
                        c1, c2, minAlpha);
  EXPECT_EQ(0,ret);
  EXPECT_NEAR(0.25,alpha,1e-8);
  EXPECT_NEAR(0,(x1 - (x0 + alpha*p)).norm(),1e-8);
  EXPECT_EQ(f1,func1(x1));
  EXPECT_LE(f1,f0 + c1*alpha*p.dot(gradx0));
  EXPECT_LE(std::fabs(p.dot(gradx1)),c2*std::fabs(p.dot(gradx0)));
}

TEST(OptimizationBFGS, rosenbrock_bfgs_convergence) {
  // -1,1 is the standard initialization for the Rosenbrock function
  std::vector<double> cont_vector(2);
  cont_vector[0] = -1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  typedef rosenbrock_model_namespace::rosenbrock_model Model;
  typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;

  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model rb_model(dummy_context);
  Optimizer bfgs(rb_model, cont_vector, disc_vector, &std::cout);

  int ret = 0;
  while (ret == 0) {
    ret = bfgs.step();
  }
  bfgs.params_r(cont_vector);

//  std::cerr << "Convergence condition: " << bfgs.get_code_string(ret) << std::endl;

  // Check that the return code is normal
  EXPECT_GE(ret,0);

  // Check the correct minimum was found
  EXPECT_NEAR(cont_vector[0],1.0,1e-6);
  EXPECT_NEAR(cont_vector[1],1.0,1e-6);

  // Check that it didn't take too long to get there
  EXPECT_LE(bfgs.iter_num(), 35);
}

TEST(OptimizationBFGS, rosenbrock_lbfgs_convergence) {
  // -1,1 is the standard initialization for the Rosenbrock function
  std::vector<double> cont_vector(2);
  cont_vector[0] = -1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  typedef rosenbrock_model_namespace::rosenbrock_model Model;
  typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::LBFGSUpdate<> > Optimizer;

  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model rb_model(dummy_context);
  Optimizer bfgs(rb_model, cont_vector, disc_vector, &std::cout);

  int ret = 0;
  while (ret == 0) {
    ret = bfgs.step();
  }
  bfgs.params_r(cont_vector);

  // Check that the return code is normal
  EXPECT_GE(ret,0);

  // Check the correct minimum was found
  EXPECT_NEAR(cont_vector[0],1.0,1e-6);
  EXPECT_NEAR(cont_vector[1],1.0,1e-6);

  // Check that it didn't take too long to get there
  EXPECT_LE(bfgs.iter_num(), 35);
}

