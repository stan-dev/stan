#include <gtest/gtest.h>
#include <stan/optimization/bfgs.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <sstream>

typedef rosenbrock_model_namespace::rosenbrock_model Model;
typedef stan::optimization::BFGSMinimizer<stan::optimization::ModelAdaptor<Model>,
                                          stan::optimization::BFGSUpdate_HInv<> > Optimizer;

class OptimizationBfgsMinimizer : public testing::Test {
public:
  Eigen::Matrix<double,Eigen::Dynamic,1> cont_vector;
  std::vector<int> disc_vector;

  void SetUp() {
    cont_vector.resize(2);
    cont_vector[0] = -1; cont_vector[1] = 1;

  }
};


TEST_F(OptimizationBfgsMinimizer, constructor) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
}

TEST_F(OptimizationBfgsMinimizer, ls_opts) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  bfgs.get_qnupdate();
}

TEST_F(OptimizationBfgsMinimizer, curr_f) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_FLOAT_EQ(bfgs.curr_f(), 4);
}

TEST_F(OptimizationBfgsMinimizer, curr_x) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);

  EXPECT_TRUE(bfgs.get_code_string(0) == "Successful step completed");
  EXPECT_TRUE(bfgs.get_code_string(10) == "Convergence detected: absolute parameter change was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(20) == "Convergence detected: absolute change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(21) == "Convergence detected: relative change in objective function was below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(30) == "Convergence detected: gradient norm is below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(31) == "Convergence detected: relative gradient magnitude is below tolerance");
  EXPECT_TRUE(bfgs.get_code_string(40) == "Maximum number of iterations hit, may not be at an optima");
  EXPECT_TRUE(bfgs.get_code_string(-1) == "Line search failed to achieve a sufficient decrease, no more progress can be made");
  EXPECT_TRUE(bfgs.get_code_string(42) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(32) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(23) == "Unknown termination code");
  EXPECT_TRUE(bfgs.get_code_string(94) == "Unknown termination code");
}

TEST_F(OptimizationBfgsMinimizer, initialize) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
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
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);
  bfgs.initialize(cont_vector);
  bfgs.step();
}

TEST_F(OptimizationBfgsMinimizer, minimize) {
  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model rb_model(dummy_context);
  std::stringstream out;
  stan::optimization::ModelAdaptor<Model> _adaptor(rb_model, disc_vector, &out);
  EXPECT_EQ("", out.str());
  Optimizer bfgs(_adaptor);

  EXPECT_FLOAT_EQ(bfgs.minimize(cont_vector), 31);
}
