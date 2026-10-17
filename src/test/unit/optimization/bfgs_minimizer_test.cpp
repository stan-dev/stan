#include <gtest/gtest.h>
#include <stan/optimization/bfgs.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <sstream>

typedef rosenbrock_model_namespace::rosenbrock_model Model;
typedef stan::optimization::BFGSMinimizer<
    stan::optimization::ModelAdaptor<Model>,
    stan::optimization::BFGSUpdate_HInv<> >
    Optimizer;

class OptimizationBfgsMinimizer : public testing::Test {
 public:
  Eigen::Matrix<double, Eigen::Dynamic, 1> cont_vector;
  std::vector<int> disc_vector;

  void SetUp() {
    cont_vector.resize(2);
    cont_vector[0] = -1;
    cont_vector[1] = 1;
  }
};

TEST_F(OptimizationBfgsMinimizer, constructor) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
}

TEST_F(OptimizationBfgsMinimizer, ls_opts) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs._ls_opts.c1, 1e-4);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.c2, 0.9);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.minAlpha, 1e-12);
  EXPECT_FLOAT_EQ(bfgs._ls_opts.alpha0, 1e-3);
}

TEST_F(OptimizationBfgsMinimizer, conv_opts) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs._conv_opts.maxIts, 10000);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.fScale, 1);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsX, 1e-8);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsF, 1e-12);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolAbsGrad, 1e-8);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolRelF, 1e+4);
  EXPECT_FLOAT_EQ(bfgs._conv_opts.tolRelGrad, 1e+3);
}

TEST_F(OptimizationBfgsMinimizer, get_qnupdate) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.get_qnupdate();
}

TEST_F(OptimizationBfgsMinimizer, curr_f) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_f(), 4);
}

TEST_F(OptimizationBfgsMinimizer, curr_x) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[1], 1);
}

TEST_F(OptimizationBfgsMinimizer, curr_g) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, curr_p) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_p()[0], 4);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_f) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_f(), 4);
}

TEST_F(OptimizationBfgsMinimizer, prev_x) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.prev_x()[1], 1);
}

TEST_F(OptimizationBfgsMinimizer, prev_g) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.prev_g()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_p) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_p()[0], 0.0040116129);
  EXPECT_FLOAT_EQ(bfgs.prev_p()[1], 0);
}

TEST_F(OptimizationBfgsMinimizer, prev_step_size) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.prev_step_size(), 0.0040000002);
}

TEST_F(OptimizationBfgsMinimizer, rel_grad_norm) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.rel_grad_norm(), 0.0012151747);
}

TEST_F(OptimizationBfgsMinimizer, rel_obj_decrease) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.rel_obj_decrease(), 0.0024023936);
}

TEST_F(OptimizationBfgsMinimizer, alpha0) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.alpha0(), 0.001);
}

TEST_F(OptimizationBfgsMinimizer, alpha) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.alpha(), 0.001);
}

TEST_F(OptimizationBfgsMinimizer, iter_num) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.iter_num(), 0);
  bfgs.step();
  EXPECT_FLOAT_EQ(bfgs.iter_num(), 1);
}

TEST_F(OptimizationBfgsMinimizer, note) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.step();
  EXPECT_TRUE(bfgs.note() == "");
}

TEST_F(OptimizationBfgsMinimizer, get_code_string) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_TRUE(bfgs.get_code_string(0) == "Successful step completed");
  EXPECT_TRUE(
      bfgs.get_code_string(10)
      == "Convergence detected: absolute parameter change was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(20) == "Convergence detected: absolute change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(21) == "Convergence detected: relative change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(30)
              == "Convergence detected: gradient norm is below tolerance");
  EXPECT_TRUE(
      bfgs.get_code_string(31)
      == "Convergence detected: relative gradient magnitude is below "
         "tolerance");
  EXPECT_TRUE(bfgs.get_code_string(40)
              == "Maximum number of iterations hit, may not be at an optima");
  EXPECT_TRUE(bfgs.get_code_string(-1) == "Line search failed to achieve a sufficient decrease, no more progress can be made");
  EXPECT_TRUE(bfgs.get_code_string(42) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(32) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(23) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(94) == "Unknown termination code");
}

TEST_F(OptimizationBfgsMinimizer, initialize) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs.curr_x().size(), 0);
  EXPECT_FLOAT_EQ(bfgs.curr_p().size(), 0);
  EXPECT_FLOAT_EQ(bfgs.curr_g().size(), 0);

  bfgs.initialize(cont_vector);
  EXPECT_FLOAT_EQ(bfgs.curr_x().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[0], -1);
  EXPECT_FLOAT_EQ(bfgs.curr_x()[1], 1);
  EXPECT_FLOAT_EQ(bfgs.curr_p().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[0], 4);
  EXPECT_FLOAT_EQ(bfgs.curr_p()[1], 0);
  EXPECT_FLOAT_EQ(bfgs.curr_g().size(), 2);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[0], -4);
  EXPECT_FLOAT_EQ(bfgs.curr_g()[1], 0);
  EXPECT_FLOAT_EQ(bfgs.iter_num(), 0);
  EXPECT_TRUE(bfgs.note() == "");
}

TEST_F(OptimizationBfgsMinimizer, step) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);
  bfgs.step();
}

TEST_F(OptimizationBfgsMinimizer, minimize) {
  stan::io::empty_var_context dummy_context;
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs.minimize(cont_vector), 31);
}

namespace {
// f(x) = sum cosh(x_i), minimised at 0; the evaluation fails where cosh
// overflows
struct minimizer_cosh {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    f = x.array().cosh().sum();
    g = x.array().sinh().matrix();
    return std::isfinite(f) && g.allFinite() ? 0 : 1;
  }
};
}  // namespace

TEST(OptimizationBfgsMinimizerCosh, large_initial_gradient) {
  // At x0 = (30, 30) the gradient is sinh(30) = 5.3e12 per coordinate. With
  // the default maxStep0 = +inf the first trial step is init_alpha * |g| =
  // 5.3e9, where cosh overflows, and after halving that step 10 times the
  // first line search fails. With maxStep0 = 1 (as the optimize services set
  // it) the first trial moves no coordinate by more than 1.
  stan::optimization::BFGSMinimizer<minimizer_cosh,
                                    stan::optimization::LBFGSUpdate<> >
      lbfgs(minimizer_cosh{});
  lbfgs._ls_opts.maxStep0 = 1;
  Eigen::VectorXd x0(2);
  x0 << 30, 30;
  lbfgs.initialize(x0);
  int ret = lbfgs.step();
  EXPECT_EQ(stan::optimization::TERM_SUCCESS, ret);
  while (ret == stan::optimization::TERM_SUCCESS) {
    ret = lbfgs.step();
  }
  EXPECT_GT(ret, 0);
  EXPECT_NEAR(2, lbfgs.curr_f(), 1e-8);
}

namespace {
// f(x) = exp(30 x) - 30 x, minimised at 0
struct minimizer_exp {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    f = std::exp(30 * x[0]) - 30 * x[0];
    g.resize(1);
    g[0] = 30 * (std::exp(30 * x[0]) - 1);
    return std::isfinite(f) && g.allFinite() ? 0 : 1;
  }
};
}  // namespace

TEST(OptimizationBfgsMinimizerExp, no_relative_convergence_after_reset) {
  // From x0 = 2 (gradient 3e27) the first line search fails with the default
  // maxStep0 = +inf. With the first trial step capped at 1 it accepts x = 1,
  // where the gradient is 3e14 and the inverse Hessian estimate from that one
  // step is 3e-28. The relative gradient test (with CmdStan's default
  // tolerance) must not stop the run there.
  stan::optimization::BFGSMinimizer<minimizer_exp,
                                    stan::optimization::LBFGSUpdate<> >
      lbfgs(minimizer_exp{});
  lbfgs._ls_opts.maxStep0 = 1;
  lbfgs._conv_opts.tolRelGrad = 1e7;
  Eigen::VectorXd x0(1);
  x0 << 2;
  lbfgs.initialize(x0);
  int ret = stan::optimization::TERM_SUCCESS;
  while (ret == stan::optimization::TERM_SUCCESS) {
    ret = lbfgs.step();
  }
  EXPECT_GT(ret, 0);
  EXPECT_NEAR(1, lbfgs.curr_f(), 1e-8);
}

namespace {
// f(x) = 1e13 x^2 / 2, minimised at 0
struct minimizer_stiff_quadratic {
  int operator()(const Eigen::VectorXd &x, double &f, Eigen::VectorXd &g) {
    f = 0.5e13 * x.squaredNorm();
    g = 1e13 * x;
    return 0;
  }
};
}  // namespace

TEST(OptimizationBfgsMinimizerStiffQuadratic, capped_step_without_floor) {
  // From x0 = 0.08 (gradient 8e11) the capped first trial step 1.25e-12
  // moves x by 1, past the minimiser at step 1e-13. That step lies below
  // minAlpha = 1e-12, the smallest step the line search may return, so the
  // first search used to fail.
  stan::optimization::BFGSMinimizer<minimizer_stiff_quadratic,
                                    stan::optimization::LBFGSUpdate<> >
      lbfgs(minimizer_stiff_quadratic{});
  lbfgs._ls_opts.maxStep0 = 1;
  Eigen::VectorXd x0(1);
  x0 << 0.08;
  lbfgs.initialize(x0);
  EXPECT_GE(lbfgs.step(), 0);
}
