#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <gtest/gtest.h>


// This is an example of how to use stan::agrad::precomputed_gradients()
TEST(StanAgradRevInternal, precomputed_gradients) {
  double value;
  std::vector<stan::agrad::var> vars;
  std::vector<double> gradients;
  stan::agrad::var x1(2), x2(3);
  stan::agrad::var y;
  
  value = 1;
  vars.resize(2);
  vars[0] = x1;
  vars[1] = x2;
  gradients.resize(2);
  gradients[0] = 4;
  gradients[1] = 5;

  EXPECT_NO_THROW(y = stan::agrad::precomputed_gradients(value, vars, gradients));
  EXPECT_FLOAT_EQ(value, y.val());

  std::vector<double> g;
  EXPECT_NO_THROW(y.grad(vars, g));
  ASSERT_EQ(2U, g.size());
  EXPECT_FLOAT_EQ(gradients[0], g[0]);
  EXPECT_FLOAT_EQ(gradients[1], g[1]);

  stan::agrad::recover_memory();
}


TEST(StanAgradRevInternal, precomputed_gradients_vari_no_independent_vars) {
  double value;
  std::vector<stan::agrad::vari *> varis;
  std::vector<double> gradients;
  
  value = 1;
  varis.resize(0);
  gradients.resize(0);
  EXPECT_NO_THROW(stan::agrad::precomputed_gradients_vari(value, varis, gradients));
  stan::agrad::precomputed_gradients_vari vari(value, varis, gradients);
  EXPECT_FLOAT_EQ(value, vari.val_);
  EXPECT_FLOAT_EQ(0, vari.adj_);
  EXPECT_NO_THROW(vari.chain());
}

TEST(StanAgradRevInternal, precomputed_gradients_vari_mismatched_sizes) {
  double value;
  std::vector<stan::agrad::vari *> varis;
  std::vector<double> gradients;

  value = 1;
  varis.resize(1);
  gradients.resize(2);
  EXPECT_THROW(stan::agrad::precomputed_gradients_vari(value, varis, gradients),
               std::invalid_argument);
}

TEST(StanAgradRevInternal, precomputed_gradients_vari) {
  double value;
  std::vector<stan::agrad::vari *> varis;
  std::vector<double> gradients;
  stan::agrad::vari x1(2), x2(3);
  
  value = 1;
  varis.resize(2);
  varis[0] = &x1;
  varis[1] = &x2;
  gradients.resize(2);
  gradients[0] = 4;
  gradients[1] = 5;
  EXPECT_NO_THROW(stan::agrad::precomputed_gradients_vari(value, varis, gradients));
  stan::agrad::precomputed_gradients_vari vari(value, varis, gradients);  
  EXPECT_FLOAT_EQ(value, vari.val_);
  EXPECT_FLOAT_EQ(0, vari.adj_);

  EXPECT_NO_THROW(vari.chain())
    << "running vari.chain() with no independent variables";
  EXPECT_FLOAT_EQ(value, vari.val_);
  EXPECT_FLOAT_EQ(0, vari.adj_);
  EXPECT_FLOAT_EQ(0, x1.adj_);
  EXPECT_FLOAT_EQ(0, x2.adj_);

  vari.init_dependent();
  EXPECT_NO_THROW(vari.chain())
    << "running vari.chain() with vari initialized as dependent variable";
  EXPECT_FLOAT_EQ(value, vari.val_);
  EXPECT_FLOAT_EQ(1, vari.adj_);
  EXPECT_FLOAT_EQ(gradients[0], x1.adj_);
  EXPECT_FLOAT_EQ(gradients[1], x2.adj_);
}

TEST(StanAgradRevInternal, precomputed_gradients_mismatched_sizes) {
  double value;
  std::vector<stan::agrad::var> vars;
  std::vector<double> gradients;
  
  value = 1;
  vars.resize(1);
  vars[0] = 0;
  gradients.resize(2);
  gradients[0] = 2;
  gradients[1] = 3;

  EXPECT_THROW(stan::agrad::precomputed_gradients(value, vars, gradients),
               std::invalid_argument);
  stan::agrad::recover_memory();
}

