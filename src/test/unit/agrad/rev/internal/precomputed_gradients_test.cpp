#include <stan/agrad/rev/internal/precomputed_gradients.hpp>
#include <gtest/gtest.h>


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
  double value = 1;
  std::vector<stan::agrad::var> vars;
  std::vector<double> gradients;

  stan::agrad::precomputed_gradients_vari vi(value, vars, gradients);
  EXPECT_FLOAT_EQ(value, vi.val_);
  EXPECT_FLOAT_EQ(0, vi.adj_);
  EXPECT_NO_THROW(vi.chain());
}

TEST(StanAgradRevInternal, precomputed_gradients_vari_mismatched_sizes) {
  double value;
  std::vector<stan::agrad::var> vars;
  std::vector<double> gradients;

  value = 1;
  vars.resize(1);
  gradients.resize(2);
  EXPECT_THROW(stan::agrad::precomputed_gradients_vari(value, vars, gradients),
               std::invalid_argument);
}

TEST(StanAgradRevInternal, precomputed_gradients_vari) {
  double value = 1;
  std::vector<stan::agrad::var> vars;
  stan::agrad::var x1(2), x2(3);
  vars.push_back(x1);
  vars.push_back(x2);

  std::vector<double> gradients;
  gradients.push_back(4);
  gradients.push_back(5);

  stan::agrad::precomputed_gradients_vari vi(value, vars, gradients);  
  EXPECT_FLOAT_EQ(value, vi.val_);
  EXPECT_FLOAT_EQ(0, vi.adj_);

  EXPECT_NO_THROW(vi.chain())
    << "running vi.chain() with no independent variables";
  EXPECT_FLOAT_EQ(value, vi.val_);
  EXPECT_FLOAT_EQ(0, vi.adj_);
  EXPECT_FLOAT_EQ(0, x1.vi_->adj_);
  EXPECT_FLOAT_EQ(0, x2.vi_->adj_);

  vi.init_dependent();
  EXPECT_NO_THROW(vi.chain())
    << "running vari.chain() with vari initialized as dependent variable";
  EXPECT_FLOAT_EQ(value, vi.val_);
  EXPECT_FLOAT_EQ(1, vi.adj_);
  EXPECT_FLOAT_EQ(gradients[0], x1.vi_->adj_);
  EXPECT_FLOAT_EQ(gradients[1], x2.vi_->adj_);
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

