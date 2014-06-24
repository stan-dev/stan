#include <gtest/gtest.h>
#include <stan/optimization/bfgs.hpp>
#include <test/test-models/no-main/optimization/rosenbrock.cpp>

TEST(OptimizationBfgs, rosenbrock_bfgs_convergence) {
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

  // Check that the return code is normal
  EXPECT_GE(ret,0);

  // Check the correct minimum was found
  EXPECT_NEAR(cont_vector[0],1.0,1e-6);
  EXPECT_NEAR(cont_vector[1],1.0,1e-6);

  // Check that it didn't take too long to get there
  EXPECT_LE(bfgs.iter_num(), 35);
  EXPECT_LE(bfgs.grad_evals(), 70);
}

TEST(OptimizationBfgs, rosenbrock_bfgs_termconds) {
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
  int ret;

  bfgs._conv_opts.maxIts = 1e9;
  bfgs._conv_opts.tolAbsX = 0;
  bfgs._conv_opts.tolAbsF = 0;
  bfgs._conv_opts.tolRelF = 0;
  bfgs._conv_opts.tolAbsGrad = 0;
  bfgs._conv_opts.tolRelGrad = 0;

  bfgs._conv_opts.maxIts = 5;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_MAXIT);
  EXPECT_EQ(bfgs.iter_num(),bfgs._conv_opts.maxIts);
  bfgs._conv_opts.maxIts = 1e9;

  bfgs._conv_opts.tolAbsX = 1e-8;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSX);
  bfgs._conv_opts.tolAbsX = 0;

  bfgs._conv_opts.tolAbsF = 1e-12;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSF);
  bfgs._conv_opts.tolAbsF = 0;

  bfgs._conv_opts.tolRelF = 1e+4;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_RELF);
  bfgs._conv_opts.tolRelF = 0;

  bfgs._conv_opts.tolAbsGrad = 1e-8;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSGRAD);
  bfgs._conv_opts.tolAbsGrad = 0;

  bfgs._conv_opts.tolRelGrad = 1e+3;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_RELGRAD);
  bfgs._conv_opts.tolRelGrad = 0;
}

TEST(OptimizationBfgs, rosenbrock_lbfgs_convergence) {
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

//  std::cerr << "Convergence condition: " << bfgs.get_code_string(ret) << std::endl;
//  std::cerr << "Converged after " << bfgs.iter_num() << " iterations and " << bfgs.grad_evals() << " gradient evaluations." << std::endl;

  // Check that the return code is normal
  EXPECT_GE(ret,0);

  // Check the correct minimum was found
  EXPECT_NEAR(cont_vector[0],1.0,1e-6);
  EXPECT_NEAR(cont_vector[1],1.0,1e-6);

  // Check that it didn't take too long to get there
  EXPECT_LE(bfgs.iter_num(), 35);
  EXPECT_LE(bfgs.grad_evals(), 70);
}

TEST(OptimizationBfgs, rosenbrock_lbfgs_termconds) {
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
  int ret;

  bfgs._conv_opts.maxIts = 1e9;
  bfgs._conv_opts.tolAbsX = 0;
  bfgs._conv_opts.tolAbsF = 0;
  bfgs._conv_opts.tolRelF = 0;
  bfgs._conv_opts.tolAbsGrad = 0;
  bfgs._conv_opts.tolRelGrad = 0;

  bfgs._conv_opts.maxIts = 5;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_MAXIT);
  EXPECT_EQ(bfgs.iter_num(),bfgs._conv_opts.maxIts);
  bfgs._conv_opts.maxIts = 1e9;

  bfgs._conv_opts.tolAbsX = 1e-8;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSX);
  bfgs._conv_opts.tolAbsX = 0;

  bfgs._conv_opts.tolAbsF = 1e-12;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSF);
  bfgs._conv_opts.tolAbsF = 0;

  bfgs._conv_opts.tolRelF = 1e+4;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_RELF);
  bfgs._conv_opts.tolRelF = 0;

  bfgs._conv_opts.tolAbsGrad = 1e-8;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_ABSGRAD);
  bfgs._conv_opts.tolAbsGrad = 0;

  bfgs._conv_opts.tolRelGrad = 1e+3;
  bfgs.initialize(cont_vector);
  while(0 == (ret = bfgs.step()));
  EXPECT_EQ(ret,stan::optimization::TERM_RELGRAD);
  bfgs._conv_opts.tolRelGrad = 0;
}
