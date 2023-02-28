#include <stan/services/pathfinder/multi.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/normal_glm.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <test/unit/services/pathfinder/util.hpp>
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
      "glm_test.data.R",
      std::fstream::in);
  return stan::io::dump(stream);
}

class ServicesPathfinderGLM : public testing::Test {
 public:
  ServicesPathfinderGLM()
      : init(init_ss),
        parameter(parameter_ss),
        diagnostics(diagnostic_ss),
        context(init_context()),
        model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::values parameter;
  stan::test::values diagnostics;
  stan::io::dump context;
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
  constexpr unsigned int seed = 0;
  constexpr unsigned int chain = 1;
  constexpr double init_radius = .7;
  constexpr double num_elbo_draws = 80;
  constexpr double num_draws = 100;
  constexpr int history_size = 15;
  constexpr double init_alpha = 1;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 0;
  constexpr double tol_rel_grad = 0;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 60;
  constexpr bool save_iterations = false;
  constexpr int num_eval_attempts = 100;
  constexpr int refresh = 1;
  stan::test::mock_callback callback;
  stan::io::empty_var_context empty_context;  // = init_init_context();
  std::ofstream empty_ostream(nullptr);
  stan::test::loggy logger(empty_ostream);

  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;

  int return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, empty_context, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      save_iterations, refresh, callback, num_elbo_draws, num_draws,
      num_eval_attempts, logger, init, parameter, diagnostics, threadpool_init);
  /*
  for (auto&& times : parameter.times_) {
    std::cout << times;
  }
  */
  // Eigen::MatrixXd param_vals = parameter.values_.transpose();
  // Eigen::MatrixXd param_vals = parameter.values_.transpose();
  Eigen::MatrixXd param_vals = std::move(parameter.values_);
  /*
  std::cout << "\n --- Optim Path ---" << std::endl;
  for (Eigen::Index i = 0; i < diagnostics.optim_path_.size(); ++i) {
    Eigen::MatrixXd tmp(2, std::get<0>(diagnostics.optim_path_[i]).size());
    tmp.row(0) = std::get<0>(diagnostics.optim_path_[i]);
    tmp.row(1) = std::get<1>(diagnostics.optim_path_[i]);
    //std::cout << "Iter: " << i << "\n" << tmp << "\n";
  }
  */
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  // std::cout << "---- Results  -------" << std::endl;
  Eigen::VectorXd mean_vals = param_vals.rowwise().mean().eval();
  //       std::cout << "Mean Values: \n" <<
  //       mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
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
  // std::cout << "\n\n" << param_vals.format(CommaInitFmt) << "\n\n";
  Eigen::MatrixXd prev_param_vals = stan::test::normal_glm_param_vals();
  // std::cout << "\n\n" << prev_param_vals.format(CommaInitFmt) << "\n\n";
  Eigen::VectorXd prev_mean_vals = prev_param_vals.rowwise().mean().eval();
  //       std::cout << "Mean Values: \n" <<
  //       mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
  Eigen::VectorXd prev_sd_vals = (((prev_param_vals.colwise() - prev_mean_vals)
                                       .array()
                                       .square()
                                       .matrix()
                                       .rowwise()
                                       .sum()
                                       .array()
                                   / (prev_param_vals.cols() - 1))
                                      .sqrt())
                                     .transpose()
                                     .eval();
  Eigen::MatrixXd ans_diff = param_vals - prev_param_vals;
  Eigen::VectorXd mean_diff_vals = ans_diff.rowwise().mean();
  //        std::cout << "diff Mean Values: \n" <<
  //        mean_diff_vals.transpose().eval().format(CommaInitFmt) << "\n";
  Eigen::VectorXd sd_diff_vals = (((ans_diff.colwise() - mean_diff_vals)
                                       .array()
                                       .square()
                                       .matrix()
                                       .rowwise()
                                       .sum()
                                       .array()
                                   / (ans_diff.cols() - 1))
                                      .sqrt())
                                     .transpose()
                                     .eval();

  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = prev_mean_vals;
  all_mean_vals.row(2) = mean_diff_vals;

  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = prev_sd_vals;
  all_sd_vals.row(2) = sd_diff_vals;
  for (int i = 0; i < all_mean_vals.cols() - 2; ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }
  for (int i = 0; i < all_mean_vals.cols() - 2; ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), .1);
  }
  // EXPECT_NEAR(0, all_mean_vals(2, 9), 0.05);
  //  std::cout << "\nAll Mean vals:\n" << all_mean_vals.format(CommaInitFmt) <<
  //  "\n"; std::cout << "\nAll SD vals:\n" << all_sd_vals.format(CommaInitFmt)
  //  << "\n"; std::cout << "\nMean vals:\n" << mean_vals.format(CommaInitFmt)
  //  << "\n";
  // std::cout << "\nSD vals:\n" << sd_vals.format(CommaInitFmt) << "\n";
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
  constexpr int num_eval_attempts = 10;

  std::ostream empty_ostream(nullptr);
  stan::test::loggy logger(empty_ostream);
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::writer> single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    //    single_path_parameter_writer.emplace_back(single_path_parameter_ss[i]);
    //    single_path_diagnostic_writer.emplace_back(single_path_diagnostic_ss[i]);
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  // int refresh = 0;
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, chain, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, save_iterations, refresh, callback, num_elbo_draws,
      num_draws, num_multi_draws, num_eval_attempts, num_paths, logger,
      std::vector<stan::callbacks::stream_writer>(num_paths, init),
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, threadpool_init);

  // Eigen::MatrixXd param_vals = parameter.values_.transpose();
  // Eigen::MatrixXd param_vals = parameter.values_.transpose();
  Eigen::MatrixXd param_vals(parameter.eigen_states_.size(),
                             parameter.eigen_states_[0].size());
  for (size_t i = 0; i < parameter.eigen_states_.size(); ++i) {
    param_vals.row(i) = parameter.eigen_states_[i];
  }
  param_vals.transposeInPlace();
  // std::cout << "\n --- Optim Path ---" << std::endl;
  for (Eigen::Index i = 0; i < diagnostics.optim_path_.size(); ++i) {
    Eigen::MatrixXd tmp(2, param_vals.cols() - 1);
    tmp.row(0) = std::get<0>(diagnostics.optim_path_[i]);
    tmp.row(1) = std::get<1>(diagnostics.optim_path_[i]);
    // std::cout << "Iter: " << i << "\n" << tmp << "\n";
  }
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  // std::cout << "---- Results  -------" << std::endl;
  Eigen::VectorXd mean_vals = param_vals.rowwise().mean().eval();
  //       std::cout << "Mean Values: \n" <<
  //       mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
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
  // std::cout << "\n\n" << param_vals.format(CommaInitFmt) << "\n\n";
  // std::cout << "rows: " << param_vals.rows() << " cols: " <<
  // param_vals.cols() << "\n";
  Eigen::MatrixXd prev_param_vals = stan::test::normal_glm_param_vals();
  // std::cout << "\n\n" << prev_param_vals.format(CommaInitFmt) << "\n\n";
  Eigen::VectorXd prev_mean_vals = prev_param_vals.rowwise().mean().eval();
  //       std::cout << "Mean Values: \n" <<
  //       mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
  Eigen::VectorXd prev_sd_vals = (((prev_param_vals.colwise() - prev_mean_vals)
                                       .array()
                                       .square()
                                       .matrix()
                                       .rowwise()
                                       .sum()
                                       .array()
                                   / (prev_param_vals.cols() - 1))
                                      .sqrt())
                                     .transpose()
                                     .eval();
  Eigen::MatrixXd ans_diff = param_vals - prev_param_vals;
  Eigen::VectorXd mean_diff_vals = ans_diff.rowwise().mean();
  //        std::cout << "diff Mean Values: \n" <<
  //        mean_diff_vals.transpose().eval().format(CommaInitFmt) << "\n";
  Eigen::VectorXd sd_diff_vals = (((ans_diff.colwise() - mean_diff_vals)
                                       .array()
                                       .square()
                                       .matrix()
                                       .rowwise()
                                       .sum()
                                       .array()
                                   / (ans_diff.cols() - 1))
                                      .sqrt())
                                     .transpose()
                                     .eval();

  Eigen::MatrixXd all_mean_vals(3, 10);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = prev_mean_vals;
  all_mean_vals.row(2) = mean_diff_vals;
  for (int i = 0; i < all_mean_vals.cols() - 2; ++i) {
    EXPECT_NEAR(0, all_mean_vals(2, i), .01);
  }

  Eigen::MatrixXd all_sd_vals(3, 10);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = prev_sd_vals;
  all_sd_vals.row(2) = sd_diff_vals;
  for (int i = 0; i < all_sd_vals.cols() - 2; ++i) {
    EXPECT_NEAR(0, all_sd_vals(2, i), 0.1);
  }
  // EXPECT_NEAR(0, all_mean_vals(2, 9), 10);
  /*
  std::cout << "\nMean vals:\n" << all_mean_vals.format(CommaInitFmt) << "\n";
  std::cout << "\nSD vals:\n" << all_sd_vals.format(CommaInitFmt) << "\n";

  std::cout << "\nMean vals:\n" << mean_vals.format(CommaInitFmt) << "\n";
  std::cout << "\nSD vals:\n" << sd_vals.format(CommaInitFmt) << "\n";
  */
}
