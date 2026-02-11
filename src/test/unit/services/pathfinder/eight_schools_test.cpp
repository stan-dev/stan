#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/math.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <test/test-models/good/services/eight_schools.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/services/util.hpp>
#include <test/unit/services/pathfinder/util.hpp>
#include <gtest/gtest.h>

// Locally tests can use threads but for jenkins we should just use 1 thread
#ifdef LOCAL_THREADS_TEST
auto&& threadpool_init = stan::math::init_threadpool_tbb(LOCAL_THREADS_TEST);
#else
auto&& threadpool_init = stan::math::init_threadpool_tbb(1);
#endif

stan::io::array_var_context init_context() {
  std::vector<std::string> names_r{"y", "sigma"};
  std::vector<double> values_r{28, 8,  -3, 7,  -1, 1,  18, 12,
                               15, 10, 16, 11, 9,  11, 10, 18};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{8}, size_vec{8}};
  std::vector<std::string> names_i{"J"};
  std::vector<int> values_i{8};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r, names_i,
                                     values_i, dims_i);
}

class ServicesPathfinderEightSchools : public testing::Test {
 public:
  ServicesPathfinderEightSchools()
      : init(init_ss),
        diagnostics(
            std::unique_ptr<std::stringstream, stan::test::deleter_noop>(
                &diagnostic_ss)),
        context(init_context()),
        model(context, 0, &model_ss) {}

  void TearDown() {
    diagnostic_ss.str(std::string());
    diagnostic_ss.clear();
    init_ss.str(std::string());
    init_ss.clear();
    model_ss.str(std::string());
    model_ss.clear();
    for (auto& ss : init_streams) {
      ss.str(std::string());
      ss.clear();
    }
    parameter.clear();
  }

  std::stringstream init_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::in_memory_writer parameter;
  stan::callbacks::json_writer<std::stringstream, stan::test::deleter_noop>
      diagnostics;
  stan::io::array_var_context context;
  stan_model model;
  static constexpr unsigned int seed = 0;
  static constexpr unsigned int stride_id = 1;
  static constexpr double init_radius = 3;
  static constexpr size_t num_multi_draws = 20000;
  static constexpr size_t num_paths = 16;
  static constexpr double num_elbo_draws = 1000;
  static constexpr double num_draws = 10000;
  static constexpr int history_size = 40;
  static constexpr double init_alpha = 1;
  static constexpr double tol_obj = 1e-12;
  static constexpr double tol_rel_obj = 1e15;
  static constexpr double tol_grad = 1e-12;
  static constexpr double tol_rel_grad = 1e15;
  static constexpr double tol_param = 1e-12;
  static constexpr int num_iterations = 2000;
  static constexpr int refresh = 1;
  static constexpr bool save_iterations = false;
  std::vector<std::stringstream> init_streams{num_paths};
  std::vector<stan::callbacks::stream_writer> init_writers{init_streams.begin(),
                                                           init_streams.end()};
};

constexpr std::array param_indices{0,  1,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
// TODO: we need to hard code this so that everything is the same between the
// two runs
auto init_init_context() { return stan::io::empty_var_context(); }

TEST_F(ServicesPathfinderEightSchools, multi) {
  // bool save_iterations = true;
  constexpr bool calculate_lp = true;
  constexpr bool resample = true;
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
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger, init_writers,
      single_path_parameter_writer, single_path_diagnostic_writer, parameter,
      diagnostics, calculate_lp, resample);

  Eigen::MatrixXd param_vals = parameter.get_eigen_state_values();
  EXPECT_EQ(param_vals.cols(), 21);
  EXPECT_EQ(param_vals.rows(), num_multi_draws);
  for (Eigen::Index i = 0; i < num_multi_draws; i++) {
    EXPECT_GE(param_vals.col(2)(i), 0);
    EXPECT_LE(param_vals.col(2)(i), num_paths);
  }
  auto param_tmp = param_vals(Eigen::indexing::all, param_indices);
  auto mean_sd_pair = stan::test::get_mean_sd(param_tmp);
  auto&& mean_vals = mean_sd_pair.first;
  auto&& sd_vals = mean_sd_pair.second;
  Eigen::RowVectorXd r_mean_vals(20);
  r_mean_vals << -17.9537, -47.016, 1.89104, 3.66449, 0.22256, 0.119645,
      -0.146812, 0.23633, -0.244868, -0.227134, 0.504507, 0.0476979, 3.66491,
      2.57979, 1.21644, 2.81399, 1.53776, 1.39865, 3.99508, 2.41488;
  Eigen::RowVectorXd r_sd_vals(20);
  r_sd_vals << 4.37932, 2.28608, 1.93964, 4.77042, 0.95799, 0.842812, 0.963455,
      0.948548, 1.03149, 0.989, 0.920778, 0.888529, 4.6405, 3.63071, 4.25895,
      4.45198, 3.90755, 4.23075, 4.56257, 4.22915;
  Eigen::MatrixXd all_mean_vals(3, 20);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = r_mean_vals;
  all_mean_vals.row(2) = mean_vals - r_mean_vals;
  // This samples badly, but is a known issue with initialization.
  for (Eigen::Index i = 0; i < all_mean_vals.cols(); i++) {
    EXPECT_NEAR(0, all_mean_vals(2, i), 1);
  }

  Eigen::MatrixXd all_sd_vals(3, 20);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = r_sd_vals;
  all_sd_vals.row(2) = sd_vals - r_sd_vals;
  for (Eigen::Index i = 0; i < all_mean_vals.cols(); i++) {
    EXPECT_NEAR(0, all_sd_vals(2, i), 2);
  }
}

TEST_F(ServicesPathfinderEightSchools, multi_psis_only_output) {
  constexpr bool calculate_lp = true;
  constexpr bool resample = true;
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  using stream_writer = stan::callbacks::unique_stream_writer<std::ofstream>;
  using string_writer
      = stan::callbacks::unique_stream_writer<std::stringstream>;
  std::vector<stream_writer> single_path_parameter_writer(num_paths);
  string_writer parameter_writer{std::make_unique<std::stringstream>(), "# "};
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger, init_writers,
      single_path_parameter_writer, single_path_diagnostic_writer,
      parameter_writer, diagnostics, calculate_lp, resample);
  string_writer parameter_writer2{std::make_unique<std::stringstream>(), "# "};
  // Check we get the same result running multiple times
  {
    std::stringstream diagnostic_ss;
    stan::callbacks::json_writer<std::stringstream, stan::test::deleter_noop>
        diagnostics{
            std::unique_ptr<std::stringstream, stan::test::deleter_noop>(
                &diagnostic_ss)};
    std::vector<std::unique_ptr<decltype(init_init_context())>>
        single_path_inits;
    for (int i = 0; i < num_paths; ++i) {
      single_path_inits.emplace_back(
          std::make_unique<decltype(init_init_context())>(init_init_context()));
    }
    std::unique_ptr<std::ostream> empty_ostream{nullptr};
    stan::test::test_logger logger(std::move(empty_ostream));
    std::vector<std::stringstream> init_streams{num_paths};
    std::vector<stan::callbacks::stream_writer> init_writers{
        init_streams.begin(), init_streams.end()};
    int return_code2 = stan::services::pathfinder::pathfinder_lbfgs_multi(
        model, single_path_inits, seed, stride_id, init_radius, history_size,
        init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
        num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
        save_iterations, refresh, callback, logger, init_writers,
        single_path_parameter_writer, single_path_diagnostic_writer,
        parameter_writer2, diagnostics, calculate_lp, resample);
    EXPECT_EQ(return_code, return_code2);
  }

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  std::stringstream tmp_stream1;
  std::stringstream tmp_stream2;
  auto&& streamer1 = parameter_writer.get_stream();
  auto&& streamer2 = parameter_writer2.get_stream();
  auto stan_data1 = stan::io::stan_csv_reader::parse(streamer1, &tmp_stream1);
  auto stan_data2 = stan::io::stan_csv_reader::parse(streamer2, &tmp_stream2);
  auto&& param_vals = stan_data1.samples;
  auto&& param_vals2 = stan_data2.samples;
  auto check_output = [](const auto& str, const auto& stan_data) {
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinders)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(PSIS)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_multi_draws);
    EXPECT_EQ(stan_data.samples.cols(), 21);
  };
  check_output(streamer1.str(), stan_data1);
  check_output(streamer2.str(), stan_data2);

  for (int j = 0; j < 21; ++j) {
    Eigen::VectorXd param_vals_col = param_vals.col(j);
    Eigen::VectorXd param_vals2_col = param_vals2.col(j);
    std::sort(param_vals_col.data(),
              param_vals_col.data() + param_vals_col.size());
    std::sort(param_vals2_col.data(),
              param_vals2_col.data() + param_vals2_col.size());
    for (Eigen::Index i = 0; i < num_multi_draws; i++) {
      EXPECT_EQ(param_vals_col(i), param_vals2_col(i))
          << "param_vals(" << i << "," << j << "): " << param_vals_col(i)
          << " != " << param_vals2_col(i);
    }
  }
}

TEST_F(ServicesPathfinderEightSchools, multi_and_single_psis_output) {
  constexpr bool calculate_lp = true;
  constexpr bool resample = true;
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  using unique_string_writer
      = stan::callbacks::unique_stream_writer<std::stringstream>;
  std::vector<unique_string_writer> single_path_parameter_writer;
  unique_string_writer parameter_writer{std::make_unique<std::stringstream>(),
                                        "# "};
  for (int i = 0; i < num_paths; ++i) {
    single_path_parameter_writer.emplace_back(
        std::make_unique<std::stringstream>(), "# ");
  }
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger, init_writers,
      single_path_parameter_writer, single_path_diagnostic_writer,
      parameter_writer, diagnostics, calculate_lp, resample);

  {
    auto&& streamer = parameter_writer.get_stream();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    auto str = streamer.str();

    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinders)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(PSIS)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_multi_draws);
    EXPECT_EQ(stan_data.samples.cols(), 21);
  }
  int sentinal = 1;
  for (auto&& single_param : single_path_parameter_writer) {
    auto&& streamer = single_param.get_stream();
    auto str = streamer.str();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinder)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_draws);
    EXPECT_EQ(stan_data.samples.cols(), 21);
    EXPECT_TRUE((stan_data.samples.col(2).array() == sentinal).all())
        << "path_id: " << stan_data.samples.col(2)(0)
        << "sentinal: " << sentinal << std::endl;
    ;
    sentinal++;
  }
}

TEST_F(ServicesPathfinderEightSchools, multi_nopsis_only_output) {
  constexpr bool calculate_lp = false;
  constexpr bool resample = false;
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  using stream_writer = stan::callbacks::unique_stream_writer<std::ofstream>;
  using string_writer
      = stan::callbacks::unique_stream_writer<std::stringstream>;
  std::vector<stream_writer> single_path_parameter_writer(num_paths);
  string_writer parameter_writer{std::make_unique<std::stringstream>(), "# "};
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger, init_writers,
      single_path_parameter_writer, single_path_diagnostic_writer,
      parameter_writer, diagnostics, calculate_lp, resample);
  auto str = parameter_writer.get_stream().str();
  {
    auto&& streamer = parameter_writer.get_stream();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinders)") == std::string::npos);
    EXPECT_TRUE(str.rfind("(PSIS)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_draws * num_paths);
    EXPECT_EQ(stan_data.samples.cols(), 21);
  }
}

TEST_F(ServicesPathfinderEightSchools, multi_and_single_nopsis_output) {
  constexpr bool calculate_lp = false;
  constexpr bool resample = false;
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  using unique_string_writer
      = stan::callbacks::unique_stream_writer<std::stringstream>;
  std::vector<unique_string_writer> single_path_parameter_writer;
  unique_string_writer parameter_writer{std::make_unique<std::stringstream>(),
                                        "# "};
  for (int i = 0; i < num_paths; ++i) {
    single_path_parameter_writer.emplace_back(
        std::make_unique<std::stringstream>(), "# ");
  }
  std::vector<stan::callbacks::json_writer<std::stringstream>>
      single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
    single_path_inits.emplace_back(
        std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, stride_id, init_radius, history_size,
      init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
      num_iterations, num_elbo_draws, num_draws, num_multi_draws, num_paths,
      save_iterations, refresh, callback, logger, init_writers,
      single_path_parameter_writer, single_path_diagnostic_writer,
      parameter_writer, diagnostics, calculate_lp, resample);

  {
    auto str = parameter_writer.get_stream().str();
    auto&& streamer = parameter_writer.get_stream();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinders)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_TRUE(str.rfind("(PSIS)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_draws * num_paths);
    EXPECT_EQ(stan_data.samples.cols(), 21);
  }
  int sentinal = 1;
  for (auto&& single_param : single_path_parameter_writer) {
    auto&& streamer = single_param.get_stream();
    auto&& str = streamer.str();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinder)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_TRUE(str.find("(PSIS)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_draws);
    EXPECT_EQ(stan_data.samples.cols(), 21);
    EXPECT_TRUE((stan_data.samples.col(2).array() == sentinal).all());
    sentinal++;
  }
}

TEST_F(ServicesPathfinderEightSchools, single_output) {
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  stan::test::mock_callback callback;
  using unique_string_writer
      = stan::callbacks::unique_stream_writer<std::stringstream>;
  unique_string_writer parameter_writer{std::make_unique<std::stringstream>(),
                                        "# "};
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, context, seed, stride_id, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      num_elbo_draws, num_draws, save_iterations, refresh, callback, logger,
      init, parameter_writer, diagnostics);
  auto str = parameter_writer.get_stream().str();
  {
    auto&& streamer = parameter_writer.get_stream();
    std::stringstream tmp_stream;
    auto stan_data = stan::io::stan_csv_reader::parse(streamer, &tmp_stream);
    EXPECT_FALSE(str.rfind("Elapsed Time:") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Pathfinder)") == std::string::npos);
    EXPECT_TRUE(str.rfind("(PSIS)") == std::string::npos);
    EXPECT_FALSE(str.rfind("(Total)") == std::string::npos);
    EXPECT_EQ(stan_data.samples.rows(), num_draws);
    EXPECT_EQ(stan_data.samples.cols(), 21);
  }
}

TEST_F(ServicesPathfinderEightSchools, single) {
  std::unique_ptr<std::ostream> empty_ostream(nullptr);
  stan::test::test_logger logger(std::move(empty_ostream));
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, context, seed, stride_id, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      num_elbo_draws, num_draws, save_iterations, refresh, callback, logger,
      init, parameter, diagnostics);

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

  Eigen::MatrixXd r_answer = stan::test::eight_schools_r_answer();

  Eigen::MatrixXd r_constrainted_draws_mat(20, 100);
  {
    stan::rng_t rng = stan::services::util::create_rng(0123, 0);
    auto fn = [&model = ServicesPathfinderEightSchools::model](auto&& u) {
      return -model.log_prob_propto_jacobian(u, 0);
    };
    Eigen::VectorXd unconstrained_draws;
    Eigen::VectorXd constrained_draws1;
    Eigen::VectorXd constrained_draws2(20);
    Eigen::VectorXd lp_approx_vec(100);
    // Results are from Lu's R code
    lp_approx_vec << -12.0415891980758, -14.6692843779338, -13.4109656242788,
        -12.227160804752, -10.8994669454787, -13.9464452858378,
        -17.7039786093493, -11.3031695577237, -12.1849838459723,
        -14.2633656680052, -13.7685697251323, -11.0849801402767,
        -10.8285877691116, -12.3078922043268, -18.4862079401751,
        -14.878979392217, -13.9884320991932, -15.7658450000531,
        -13.5906482194447, -12.9120430284407, -18.2651279783073,
        -13.0161106634425, -14.6633050842275, -15.708171891455,
        -13.8002820377402, -13.4484536964903, -12.9558192824891,
        -18.030159468489, -12.436042490926, -12.7938205793498,
        -15.4295215357008, -11.7361108739125, -14.1692223330973,
        -12.4698540687768, -16.2225112479695, -14.6021099557893,
        -15.4163482862364, -11.9367428966647, -15.6987363918049,
        -13.2541127046878, -13.395247477582, -13.7297660475934,
        -15.5881489265056, -13.5906575138153, -19.5817805593569,
        -15.3874299612537, -14.7803838914721, -13.5453155677371,
        -18.5256438441971, -21.7907055918946, -13.9876362902857,
        -14.3584339685507, -12.3086782261963, -13.4520009680182,
        -13.2565205387879, -14.8449352555917, -11.7995060730947,
        -16.1673766763038, -13.8230070576965, -14.4323461406136,
        -14.5139646362747, -15.7152727007162, -16.0978882701874,
        -12.8437110780737, -16.1267323384854, -17.5695117515445,
        -15.7244669033694, -14.318592510172, -13.6331931944301,
        -15.3973326320899, -16.6577158373945, -17.0600363400148,
        -13.3516348546988, -12.2942663317071, -19.1148011460955,
        -17.6392635944591, -13.3379766819778, -13.8803098238232,
        -12.5059777414601, -15.8823434809178, -14.5040005356544,
        -17.9707192175747, -14.3296312988667, -15.9246135209721,
        -20.6431707513941, -14.2483182078639, -12.9012691966467,
        -11.8312105455114, -14.2360469104402, -14.1732053430172,
        -12.7669225560584, -14.3443242235104, -14.4185150275073,
        -16.9557240942739, -14.2902638224899, -13.2814736915503,
        -20.7083049704887, -17.6192198763631, -10.705036567492,
        -12.1087056948567;
    for (Eigen::Index i = 0; i < r_answer.cols(); ++i) {
      unconstrained_draws = r_answer.col(i);
      model.write_array(rng, unconstrained_draws, constrained_draws1);
      constrained_draws2.tail(18) = constrained_draws1;
      constrained_draws2(0) = lp_approx_vec(i);
      constrained_draws2(1) = -fn(unconstrained_draws);
      r_constrainted_draws_mat.col(i) = constrained_draws2;
    }
  }
  Eigen::RowVectorXd mean_r_vals
      = r_constrainted_draws_mat.rowwise().mean().transpose();
  Eigen::RowVectorXd sd_r_vals
      = (((r_constrainted_draws_mat.colwise() - mean_r_vals.transpose())
              .array()
              .square()
              .matrix()
              .rowwise()
              .sum()
              .array()
          / (r_constrainted_draws_mat.cols() - 1))
             .sqrt())
            .transpose()
            .eval();

  Eigen::MatrixXd all_mean_vals(3, 20);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = mean_r_vals;
  all_mean_vals.row(2) = mean_vals - mean_r_vals;
  Eigen::MatrixXd all_sd_vals(3, 20);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = sd_r_vals;
  all_sd_vals.row(2) = sd_vals - sd_r_vals;

  // Single pathfinder can do very badly for eight schools
  for (Eigen::Index i = 2; i < all_mean_vals.cols(); i++) {
    EXPECT_NEAR(0, all_mean_vals(2, i), 3);
  }

  for (Eigen::Index i = 2; i < all_sd_vals.cols(); i++) {
    EXPECT_NEAR(0, all_sd_vals(2, i), 6);
  }
}
