#include <gtest/gtest.h>
#include <stan/optimization/bfgs.hpp>
#include <test/test-models/good/optimization/exponential_boundary.hpp>

typedef exponential_boundary_model_namespace::exponential_boundary_model Model;
typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;

TEST(OptimizationBfgs, exponential_boundary_nonconvergence) {
  std::vector<double> cont_vector(2);
  cont_vector[0] = 1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model eb_model(dummy_context);
  std::stringstream out;
  Optimizer bfgs(eb_model, cont_vector, disc_vector, &out);
  EXPECT_EQ("", out.str());

  int ret = 0;
  while (ret == 0) {
    ret = bfgs.step();
  }
  bfgs.params_r(cont_vector);

  // Check that line search failed: TERM_LSFAIL = -1;
  EXPECT_LE(ret, -1);
}
