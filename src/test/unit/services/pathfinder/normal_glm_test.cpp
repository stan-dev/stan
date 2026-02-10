#include <stdexcept>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <test/test-models/good/services/normal_glm.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/services/pathfinder/util.hpp>
#include <test/unit/services/util.hpp>
#include <test/unit/util.hpp>
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
        parameter(),
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

  std::stringstream init_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::in_memory_writer parameter;
  stan::callbacks::json_writer<std::stringstream, stan::test::deleter_noop>
      diagnostics;
  stan::json::json_data context;
  stan_model model;
};
constexpr std::array param_indices{0, 1, 3, 4, 5, 6, 7, 8, 9, 10};
inline stan::io::array_var_context init_init_context() {
  std::vector<std::string> names_r{};
  std::vector<double> values_r{};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{};
  std::vector<std::string> names_i{""};
  std::vector<int> values_i{};
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r);
}

TEST_F(ServicesPathfinderGLM, single) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_elbo_draws = 80;
  constexpr double num_draws = 500;
  constexpr int history_size = 35;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 400;
  constexpr bool save_iterations = true;
  constexpr int refresh = 1;
  stan::test::mock_callback callback;
  stan::io::array_var_context init_context = init_init_context();
  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));

  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;

  int rc = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, init_context, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, save_iterations, refresh,
      callback, logger, init, parameter, diagnostics);
  ASSERT_EQ(rc, 0);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  for (auto&& x_i : param_vals.col(2)) {
    EXPECT_EQ(x_i, stride_id);
  }

  auto param_tmp = param_vals(Eigen::indexing::all, param_indices);
  auto mean_sd_pair = stan::test::get_mean_sd(param_tmp);
  auto&& mean_vals = mean_sd_pair.first;
  auto&& sd_vals = mean_sd_pair.second;
  auto prev_param_summary = stan::test::normal_glm_param_summary();
  Eigen::Matrix<double, 1, 10> prev_mean_vals = prev_param_summary.first;
  Eigen::Matrix<double, 1, 10> prev_sd_vals = prev_param_summary.second;
  Eigen::RowVectorXd ans_mean_diff = mean_vals - prev_mean_vals;
  Eigen::RowVectorXd ans_sd_diff = sd_vals - prev_sd_vals;
  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = prev_mean_vals;
  all_mean_vals.row(2) = ans_mean_diff;
  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = prev_sd_vals;
  all_sd_vals.row(2) = ans_sd_diff;
  // True Sd's are all 1 and true means are -4, -2, 0, 1, 3, -1
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), .1);
  }
  auto json = diagnostic_ss.str();
  ASSERT_TRUE(stan::test::is_valid_JSON(json));
}

TEST_F(ServicesPathfinderGLM, single_noreturnlp) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_elbo_draws = 80;
  constexpr double num_draws = 500;
  constexpr int history_size = 35;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 400;
  constexpr bool save_iterations = true;
  constexpr int refresh = 1;
  constexpr bool calculate_lp = false;

  stan::test::mock_callback callback;
  stan::io::array_var_context init_context = init_init_context();
  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));

  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;

  int rc = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, init_context, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, save_iterations, refresh,
      callback, logger, init, parameter, diagnostics, calculate_lp);
  ASSERT_EQ(rc, 0);
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(11, param_vals.cols());
  EXPECT_EQ(500, param_vals.rows());
  for (auto&& x_i : param_vals.col(2)) {
    EXPECT_EQ(x_i, stride_id);
  }
  for (Eigen::Index i = 0; i < num_elbo_draws; ++i) {
    EXPECT_FALSE(std::isnan(param_vals.coeff(num_draws + i, 1)))
        << "row: " << (num_draws + i);
  }
  for (Eigen::Index i = 0; i < (num_draws - num_elbo_draws); ++i) {
    EXPECT_TRUE(std::isnan(param_vals.coeff(num_elbo_draws + i, 1)))
        << "row: " << (num_draws + num_elbo_draws + i);
  }
}

namespace stan::test {
template <typename T>
void init_null_writers(std::vector<T>& writers, size_t num_chains) {
  writers.reserve(num_chains);
  for (size_t i = 0; i < num_chains; ++i) {
    writers.emplace_back(nullptr);
  }
}
}  // namespace stan::test

TEST_F(ServicesPathfinderGLM, multi_null_unique) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 1000;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  constexpr int calculate_lp = true;
  constexpr int resample = true;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::unique_stream_writer<std::ofstream>>
      single_path_parameter_writer;
  stan::test::init_null_writers(single_path_parameter_writer, num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  using init_context_t = decltype(init_init_context());
  std::vector<std::unique_ptr<init_context_t>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<init_context_t>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(11, param_vals.cols());
  EXPECT_EQ(1000, param_vals.rows());
  // They can be in any order and any number
  for (Eigen::Index i = 0; i < num_multi_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
  auto param_tmp = param_vals(Eigen::indexing::all, param_indices);
  auto mean_sd_pair = stan::test::get_mean_sd(param_tmp);
  auto&& mean_vals = mean_sd_pair.first;
  auto&& sd_vals = mean_sd_pair.second;
  auto prev_param_summary = stan::test::normal_glm_param_summary();
  Eigen::Matrix<double, 1, 10> prev_mean_vals = prev_param_summary.first;
  Eigen::Matrix<double, 1, 10> prev_sd_vals = prev_param_summary.second;
  Eigen::RowVectorXd ans_mean_diff = mean_vals - prev_mean_vals;
  Eigen::RowVectorXd ans_sd_diff = sd_vals - prev_sd_vals;
  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = prev_mean_vals;
  all_mean_vals.row(2) = ans_mean_diff;
  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = prev_sd_vals;
  all_sd_vals.row(2) = ans_sd_diff;
  // True Sd's are all 1 and true means are -4, -2, 0, 1, 3, -1
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), 1e-2);
  }
}

TEST_F(ServicesPathfinderGLM, multi) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 1000;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  constexpr int calculate_lp = true;
  constexpr int resample = true;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  using init_context_t = decltype(init_init_context());
  std::vector<std::unique_ptr<init_context_t>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<init_context_t>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(11, param_vals.cols());
  EXPECT_EQ(1000, param_vals.rows());
  // They can be in any order and any number
  for (Eigen::Index i = 0; i < num_multi_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
  auto param_tmp = param_vals(Eigen::indexing::all, param_indices);
  auto mean_sd_pair = stan::test::get_mean_sd(param_tmp);
  auto&& mean_vals = mean_sd_pair.first;
  auto&& sd_vals = mean_sd_pair.second;
  auto prev_param_summary = stan::test::normal_glm_param_summary();
  Eigen::Matrix<double, 1, 10> prev_mean_vals = prev_param_summary.first;
  Eigen::Matrix<double, 1, 10> prev_sd_vals = prev_param_summary.second;
  Eigen::RowVectorXd ans_mean_diff = mean_vals - prev_mean_vals;
  Eigen::RowVectorXd ans_sd_diff = sd_vals - prev_sd_vals;
  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = prev_mean_vals;
  all_mean_vals.row(2) = ans_mean_diff;
  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = prev_sd_vals;
  all_sd_vals.row(2) = ans_sd_diff;
  // True Sd's are all 1 and true means are -4, -2, 0, 1, 3, -1
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 2; i < all_mean_vals.cols(); ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), 1e-2);
  }
}

TEST_F(ServicesPathfinderGLM, multi_noresample) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 100;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  // Should return num_paths * num_draws = 8000
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  constexpr bool calculate_lp = true;
  constexpr bool resample = false;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(11, param_vals.cols());
  EXPECT_EQ(8000, param_vals.rows());
  for (Eigen::Index i = 0; i < num_multi_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
}

TEST_F(ServicesPathfinderGLM, multi_noresample_noreturnlp) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 100;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 10;
  // Should return num_paths * num_draws = 8000
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  constexpr bool calculate_lp = false;
  constexpr bool resample = false;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  EXPECT_EQ(param_vals.cols(), 11);
  EXPECT_EQ(param_vals.rows(),
            8000);  // They can be in any order and any number
  for (Eigen::Index i = 0; i < num_multi_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }

  // Parallel means we don't know order
  bool is_all_lp = true;
  bool is_any_lp = false;
  for (Eigen::Index i = 0; i < num_draws * num_paths; ++i) {
    is_all_lp &= std::isnan(param_vals.coeff(i, 1));
    is_any_lp |= !std::isnan(param_vals.coeff(i, 1));
  }
  EXPECT_FALSE(is_all_lp);
  EXPECT_TRUE(is_any_lp);
}

TEST_F(ServicesPathfinderGLM, multi_resample_noreturnlp) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 100;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  // Should return num_paths * num_draws = 8000
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  //
  constexpr bool calculate_lp = false;
  constexpr bool resample = true;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  EXPECT_EQ(param_vals.cols(), 11);
  EXPECT_EQ(param_vals.rows(), 8000);
  // They can be in any order and any number
  for (Eigen::Index i = 0; i < num_paths * num_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
  bool is_all_lp = true;
  bool is_any_lp = false;
  for (Eigen::Index i = 0; i < num_draws * num_paths; ++i) {
    is_all_lp &= std::isnan(param_vals.coeff(i, 1));
    is_any_lp |= !std::isnan(param_vals.coeff(i, 1));
  }
  EXPECT_FALSE(is_all_lp);
  EXPECT_TRUE(is_any_lp);
}

TEST_F(ServicesPathfinderGLM, multi_noresample_returnlp) {
  constexpr unsigned int seed = 3;
  constexpr unsigned int stride_id = 1;
  constexpr double init_radius = 0.5;
  constexpr double num_multi_draws = 100;
  constexpr int num_paths = 4;
  constexpr double num_elbo_draws = 1000;
  // Should return num_paths * num_draws = 8000
  constexpr double num_draws = 2000;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 2e-4;
  constexpr double tol_rel_grad = 2e-6;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 220;
  constexpr bool save_iterations = false;
  constexpr int refresh = 0;
  constexpr bool calculate_lp = true;
  constexpr bool resample = false;

  std::unique_ptr<std::stringstream> string_ostream(new std::stringstream{});
  stan::test::test_logger logger(std::move(string_ostream));
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int rc = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);
  ASSERT_EQ(rc, 0);
  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(param_vals.cols(), 11);
  EXPECT_EQ(param_vals.rows(),
            8000);  // They can be in any order and any number
  for (Eigen::Index i = 0; i < num_paths * num_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  bool is_all_lp = true;
  bool is_any_lp = false;
  for (Eigen::Index i = 0; i < num_draws * num_paths; ++i) {
    is_all_lp &= std::isnan(param_vals.coeff(i, 1));
    is_any_lp |= !std::isnan(param_vals.coeff(i, 1));
  }
  EXPECT_FALSE(is_all_lp);
  EXPECT_TRUE(is_any_lp);
}
