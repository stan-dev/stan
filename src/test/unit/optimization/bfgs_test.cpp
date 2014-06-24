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

TEST(OptimizationBfgs, ConvergenceOptions) {
  FAIL() 
    << "add tests as a means of fixing behavior";
}

TEST(OptimizationBfgs, LsOptions) {
  FAIL() 
    << "add tests as a means of fixing behavior";
}

TEST(OptimizationBfgs, BfgsMinimizer) {
  FAIL()
    << "add tests for class BFGSMinimizer (construction)";
}

TEST(OptimizationBfgs, BfgsMinimizer__ls_opts) {
  FAIL()
    << "add tests for class BFGSMinimizer._ls_opts";
}

TEST(OptimizationBfgs, BfgsMinimizer__conv_opts) {
  FAIL()
    << "add tests for class BFGSMinimizer._conv_opts";
}

TEST(OptimizationBfgs, BfgsMinimizer_get_qnupdate) {
  FAIL()
    << "add tests for class BFGSMinimizer.get_qnupdate()";
}

TEST(OptimizationBfgs, BfgsMinimizer_curr_f) {
  FAIL()
    << "add tests for class BFGSMinimizer.curr_f()";
}

TEST(OptimizationBfgs, BfgsMinimizer_curr_x) {
  FAIL()
    << "add tests for class BFGSMinimizer.curr_x()";
}

TEST(OptimizationBfgs, BfgsMinimizer_curr_g) {
  FAIL()
    << "add tests for class BFGSMinimizer.curr_g()";
}

TEST(OptimizationBfgs, BfgsMinimizer_curr_p) {
  FAIL()
    << "add tests for class BFGSMinimizer.curr_p()";
}

TEST(OptimizationBfgs, BfgsMinimizer_prev_f) {
  FAIL()
    << "add tests for class BFGSMinimizer.prev_f()";
}

TEST(OptimizationBfgs, BfgsMinimizer_prev_x) {
  FAIL()
    << "add tests for class BFGSMinimizer.prev_x()";
}

TEST(OptimizationBfgs, BfgsMinimizer_prev_g) {
  FAIL()
    << "add tests for class BFGSMinimizer.prev_g()";
}

TEST(OptimizationBfgs, BfgsMinimizer_prev_p) {
  FAIL()
    << "add tests for class BFGSMinimizer.prev_p()";
}

TEST(OptimizationBfgs, BfgsMinimizer_prev_step_size) {
  FAIL()
    << "add tests for class BFGSMinimizer.prev_step_size()";
}

TEST(OptimizationBfgs, BfgsMinimizer_rel_grad_norm) {
  FAIL()
    << "add tests for class BFGSMinimizer.rel_grad_norm()";
}

TEST(OptimizationBfgs, BfgsMinimizer_rel_obj_decrease) {
  FAIL()
    << "add tests for class BFGSMinimizer.rel_grad_norm()";
}

TEST(OptimizationBfgs, BfgsMinimizer_alpha0) {
  FAIL()
    << "add tests for class BFGSMinimizer.alpha0()";
}

TEST(OptimizationBfgs, BfgsMinimizer_alpha) {
  FAIL()
    << "add tests for class BFGSMinimizer.alpha()";
}

TEST(OptimizationBfgs, BfgsMinimizer_iter_num) {
  FAIL()
    << "add tests for class BFGSMinimizer.iter_num()";
}

TEST(OptimizationBfgs, BfgsMinimizer_note) {
  FAIL()
    << "add tests for class BFGSMinimizer.note()";
}

TEST(OptimizationBfgs, BfgsMinimizer_get_code_string) {
  FAIL()
    << "add tests for class BFGSMinimizer.get_code_string()";
}

TEST(OptimizationBfgs, BfgsMinimizer_initialize) {
  FAIL()
    << "add tests for class BFGSMinimizer.initialize()";
}

TEST(OptimizationBfgs, BfgsMinimizer_step) {
  FAIL()
    << "add tests for class BFGSMinimizer.step()";
}

TEST(OptimizationBfgs, BfgsMinimizer_minimize) {
  FAIL()
    << "add tests for class BFGSMinimizer.minimize()";
}


TEST(OptimizationBfgs, lp_no_jacobian) {
  FAIL()
    << "add tests for lp_no_jacobian() <- is this not used and should it be removed?";
}

TEST(OptimizationBfgs, ModelAdaptor) {
  FAIL() 
    << "add tests for ModelAdaptor (construction)";
}

TEST(OptimizationBfgs, ModelAdaptor_fevals) {
  FAIL() 
    << "add tests for ModelAdaptor.fevals()";
}

TEST(OptimizationBfgs, ModelAdaptor_operator_parens__matrix_double) {
  FAIL() 
    << "add tests for ModelAdaptor.operator(Eigen::Matrix, double)";
}

TEST(OptimizationBfgs, ModelAdaptor_operator_parens__matrix_double_matrix) {
  FAIL() 
    << "add tests for ModelAdaptor.operator(Eigen::Matrix, double, Eigen::Matrix)";
}

TEST(OptimizationBfgs, ModelAdaptor_df) {
  FAIL() 
    << "add tests for ModelAdaptor.fevals()";
}

TEST(OptimizationBfgs, BFGSLineSearch) {
  FAIL() 
    << "add tests for BFGSLineSearch (construction) -- see tests above";
}


TEST(OptimizationBfgs, BFGSLineSearch_initialize) {
  FAIL() 
    << "add tests for BFGSLineSearch.initialize()";
}

TEST(OptimizationBfgs, BFGSLineSearch_grad_evals) {
  FAIL() 
    << "add tests for BFGSLineSearch.grad_evals()";
}

TEST(OptimizationBfgs, BFGSLineSearch_logp) {
  FAIL() 
    << "add tests for BFGSLineSearch.logp()";
}

TEST(OptimizationBfgs, BFGSLineSearch_grad_norm) {
  FAIL() 
    << "add tests for BFGSLineSearch.grad_norm()";
}

TEST(OptimizationBfgs, BFGSLineSearch_grad) {
  FAIL() 
    << "add tests for BFGSLineSearch.grad()";
}

TEST(OptimizationBfgs, BFGSLineSearch_params_r) {
  FAIL() 
    << "add tests for BFGSLineSearch.params_r()";
}
