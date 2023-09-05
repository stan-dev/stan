#include <stdexcept>
#include <stan/callbacks/json_writer.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <test/test-models/good/services/normal_glm.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/services/pathfinder/util.hpp>
#include <test/unit/services/util.hpp>
#include <rapidjson/document.h>
#include <gtest/gtest.h>

// Locally tests can use threads but for jenkins we should just use 1 thread
#ifdef LOCAL_THREADS_TEST
auto&& threadpool_init = stan::math::init_threadpool_tbb(LOCAL_THREADS_TEST);
#else
auto&& threadpool_init = stan::math::init_threadpool_tbb(1);
#endif

auto init_context() {
  std::fstream stream(
      "./src/test/unit/services/pathfinder/"
      "normal_glm_test.json",
      std::fstream::in);
  return stan::json::json_data(stream);
}

class ServicesPathfinderGLM : public testing::Test {
 public:
  ServicesPathfinderGLM()
      : init(init_ss),
        parameter(parameter_ss),
        diagnostics(
            std::unique_ptr<std::stringstream, stan::test::deleter_noop>(
                &diagnostic_ss)),
        context(init_context()),
        model(context, 0, &model_ss) {}

  void SetUp() {
    diagnostic_ss.str(std::string());
    diagnostic_ss.clear();
  }
  void TearDown() {}

  std::stringstream init_ss, parameter_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::in_memory_writer parameter;
  stan::callbacks::json_writer<std::stringstream, stan::test::deleter_noop>
      diagnostics;
  stan::json::json_data context;
  stan_model model;
};

stan::io::array_var_context init_init_context() {
  std::vector<std::string> names_r{"b", "Intercept", "sigma"};
  std::vector<double> values_r{0, 0, 0, 0, 0, 0, 1};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{5}, size_vec{}, size_vec{}};
  std::vector<std::string> names_i{""};
  std::vector<int> values_i{};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r);
}

TEST_F(ServicesPathfinderGLM, single) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int chain = 1;
  constexpr double init_radius = 2;
  constexpr double num_elbo_draws = 80;
  constexpr double num_draws = 500;
  constexpr int history_size = 35;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 0;
  constexpr double tol_rel_grad = 0;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 400;
  constexpr bool save_iterations = true;
  constexpr int refresh = 1;

  stan::test::mock_callback callback;
  stan::io::empty_var_context empty_context;  // = init_init_context();
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));

  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;

  int return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, empty_context, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      num_elbo_draws, num_draws, save_iterations, refresh, callback, logger,
      init, parameter, diagnostics);
  Eigen::MatrixXd param_vals = std::move(parameter.values_);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  Eigen::VectorXd mean_vals = param_vals.rowwise().mean().eval();
  Eigen::VectorXd sd_vals = (((param_vals.colwise() - mean_vals)
                                  .array()
                                  .square()
                                  .matrix()
                                  .rowwise()
                                  .sum()
                                  .array()
                              / (param_vals.cols() - 1))
                                 .sqrt())
                                .transpose()
                                .eval();

  auto prev_param_summary = stan::test::normal_glm_param_summary();
  Eigen::VectorXd prev_mean_vals = prev_param_summary.first;
  Eigen::VectorXd prev_sd_vals = prev_param_summary.second;
  Eigen::MatrixXd ans_mean_diff = mean_vals - prev_mean_vals;
  Eigen::MatrixXd ans_sd_diff = sd_vals - prev_sd_vals;
  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals.transpose();
  all_mean_vals.row(1) = prev_mean_vals.transpose();
  all_mean_vals.row(2) = ans_mean_diff.transpose();
  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals.transpose();
  all_sd_vals.row(1) = prev_sd_vals.transpose();
  all_sd_vals.row(2) = ans_sd_diff.transpose();
  // True Sd's are all 1 and true means are -4, -2, 0, 1, 3, -1
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), .1);
  }
  rapidjson::Document document;
  ASSERT_FALSE(document.Parse<0>(diagnostic_ss.str().c_str()).HasParseError());
}

TEST_F(ServicesPathfinderGLM, multi) {
  constexpr unsigned int seed = 0;
  constexpr unsigned int chain = 1;
  constexpr double init_radius = 1;
  constexpr double num_multi_draws = 100;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 0;
  constexpr double tol_rel_grad = 0;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;

  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, chain, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics);

  Eigen::MatrixXd param_vals(parameter.eigen_states_.size(),
                             parameter.eigen_states_[0].size());
  for (size_t i = 0; i < parameter.eigen_states_.size(); ++i) {
    param_vals.row(i) = parameter.eigen_states_[i];
  }
  param_vals.transposeInPlace();
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  Eigen::VectorXd mean_vals = param_vals.rowwise().mean().eval();
  Eigen::VectorXd sd_vals = (((param_vals.colwise() - mean_vals)
                                  .array()
                                  .square()
                                  .matrix()
                                  .rowwise()
                                  .sum()
                                  .array()
                              / (param_vals.cols() - 1))
                                 .sqrt())
                                .transpose()
                                .eval();
  auto prev_param_summary = stan::test::normal_glm_param_summary();
  Eigen::VectorXd prev_mean_vals = prev_param_summary.first;
  Eigen::VectorXd prev_sd_vals = prev_param_summary.second;
  Eigen::MatrixXd ans_mean_diff = mean_vals - prev_mean_vals;
  Eigen::MatrixXd ans_sd_diff = sd_vals - prev_sd_vals;
  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals.transpose();
  all_mean_vals.row(1) = prev_mean_vals.transpose();
  all_mean_vals.row(2) = ans_mean_diff.transpose();
  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals.transpose();
  all_sd_vals.row(1) = prev_sd_vals.transpose();
  all_sd_vals.row(2) = ans_sd_diff.transpose();
  // True Sd's are all 1 and true means are -4, -2, 0, 1, 3, -1
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), .1);
  }
}
