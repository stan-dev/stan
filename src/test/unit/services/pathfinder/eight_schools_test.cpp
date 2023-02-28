#include <stan/math.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/eight_schools.hpp>
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
        parameter(parameter_ss),
        diagnostics(diagnostic_ss),
        context(init_context()),
        model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::values parameter;
  stan::test::values diagnostics;
  stan::io::array_var_context context;
  stan_model model;
};

auto init_init_context() { return stan::io::empty_var_context(); }

TEST_F(ServicesPathfinderEightSchools, multi) {
  constexpr unsigned int seed = 0;
  constexpr unsigned int chain = 1;
  constexpr double init_radius = .6;
  constexpr size_t num_multi_draws = 10000;
  constexpr size_t num_paths = 16;
  constexpr double num_elbo_draws = 1000;
  constexpr double num_draws = 10000;
  constexpr int history_size = 10;
  constexpr double init_alpha = 0.01;
  constexpr double tol_obj = 0;
  constexpr double tol_rel_obj = 0;
  constexpr double tol_grad = 0;
  constexpr double tol_rel_grad = 0;
  constexpr double tol_param = 0;
  constexpr int num_iterations = 500;
  constexpr int num_eval_attempts = 20;
  // bool save_iterations = true;
  constexpr int refresh = 0;
  constexpr bool save_iterations = false;
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
  Eigen::MatrixXd param_vals(parameter.eigen_states_.size(),
                             parameter.eigen_states_[0].size());
  for (size_t i = 0; i < parameter.eigen_states_.size(); ++i) {
    param_vals.row(i) = parameter.eigen_states_[i];
  }

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  // std::cout << "Values: \n" << param_vals.format(CommaInitFmt) << "\n";

  Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
  Eigen::RowVectorXd sd_vals = ((param_vals.rowwise() - mean_vals)
                                    .array()
                                    .square()
                                    .matrix()
                                    .colwise()
                                    .sum()
                                    .array()
                                / (param_vals.rows() - 1))
                                   .sqrt();

  // std::cout << "Mean Values: \n" << mean_vals.format(CommaInitFmt) << "\n";
  // std::cout << "SD Values: \n" << sd_vals.format(CommaInitFmt) << "\n";

  Eigen::RowVectorXd prev_mean_vals(20);
  prev_mean_vals << 1.89104, 3.66449, 0.22256, 0.119645, -0.146812, 0.23633,
      -0.244868, -0.227134, 0.504507, 0.0476979, 3.66491, 2.57979, 1.21644,
      2.81399, 1.53776, 1.39865, 3.99508, 2.41488, -17.9537, -47.016;
  Eigen::RowVectorXd prev_sd_vals(20);
  prev_sd_vals << 1.93964, 4.77042, 0.95799, 0.842812, 0.963455, 0.948548,
      1.03149, 0.989, 0.920778, 0.888529, 4.6405, 3.63071, 4.25895, 4.45198,
      3.90755, 4.23075, 4.56257, 4.22915, 4.37932, 2.28608;
  Eigen::VectorXd diff_vals(20);
  diff_vals = ((prev_mean_vals.array() + (2.0 * prev_sd_vals.array())).abs()
               - mean_vals.array().abs())
                  .matrix()
                  .transpose();
  for (Eigen::Index i = 0; i < 18; i++) {
    EXPECT_GE(diff_vals(i), 0.0);
  }

  // std::cout << "Diffs\n" << diff_vals.format(CommaInitFmt) << "\n";
}

TEST_F(ServicesPathfinderEightSchools, single) {
  constexpr unsigned int seed = 0;
  constexpr unsigned int chain = 1;
  constexpr double init_radius = .7;
  constexpr double num_elbo_draws = 80;
  constexpr double num_draws = 100;
  constexpr int history_size = 6;
  constexpr double init_alpha = 0.001;
  constexpr double tol_obj = 1e-12;
  constexpr double tol_rel_obj = 10000;
  constexpr double tol_grad = 1e-8;
  constexpr double tol_rel_grad = 10000000;
  constexpr double tol_param = 1e-8;
  constexpr int num_iterations = 2000;
  constexpr bool save_iterations = false;
  constexpr int num_eval_attempts = 100;
  constexpr int refresh = 0;
  std::ostream empty_ostream(nullptr);
  stan::test::loggy logger(empty_ostream);
  stan::test::mock_callback callback;
  int return_code = stan::services::pathfinder::pathfinder_lbfgs_single(
      model, context, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      save_iterations, refresh, callback, num_elbo_draws, num_draws,
      num_eval_attempts, logger, init, parameter, diagnostics, threadpool_init);

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  Eigen::MatrixXd param_vals = std::move(parameter.values_);

  // std::cout << "---- Results  -------" << std::endl;
  /*
  std::cout << "Values: \n"
            << param_vals.format(CommaInitFmt) << "\n";
  */
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

  Eigen::MatrixXd r_answer = stan::test::eight_schools_r_answer();

  Eigen::MatrixXd r_constrainted_draws_mat(20, 100);
  {
    auto rng = stan::services::util::create_rng<boost::ecuyer1988>(0123, 0);
    auto fn = [&model = ServicesPathfinderEightSchools::model](auto&& u) {
      return -model.log_prob_propto_jacobian(u, 0);
    };
    Eigen::VectorXd unconstrained_draws;
    Eigen::VectorXd constrained_draws1;
    Eigen::VectorXd constrained_draws2(20);
    Eigen::VectorXd lp_approx_vec(100);
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
      constrained_draws2.head(18) = constrained_draws1;
      constrained_draws2(18) = lp_approx_vec(i);
      constrained_draws2(19) = -fn(unconstrained_draws);
      r_constrainted_draws_mat.col(i) = constrained_draws2;
    }
  }
  Eigen::VectorXd mean_r_vals = r_constrainted_draws_mat.rowwise().mean();
  //        std::cout << "R Mean Values: \n" <<
  //        mean_r_vals.transpose().eval().format(CommaInitFmt) << "\n";
  Eigen::VectorXd sd_r_vals
      = (((r_constrainted_draws_mat.colwise() - mean_r_vals)
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
  Eigen::VectorXd diff_vals(20);
  diff_vals = ((mean_r_vals.array() + (2.0 * sd_r_vals.array())).abs()
               - mean_vals.array().abs())
                  .matrix();
  for (Eigen::Index i = 0; i < 18; i++) {
    EXPECT_GE(diff_vals(i), 0.0);
  }

  Eigen::MatrixXd ans_diff = param_vals - r_constrainted_draws_mat;
  Eigen::VectorXd mean_diff_vals = ans_diff.rowwise().mean();
  //      std::cout << "diff Mean Values: \n" <<
  //      mean_diff_vals.transpose().eval().format(CommaInitFmt) << "\n";
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

  Eigen::MatrixXd all_mean_vals(3, 20);
  all_mean_vals.row(0) = mean_vals;
  all_mean_vals.row(1) = mean_r_vals;
  all_mean_vals.row(2) = mean_diff_vals;

  Eigen::MatrixXd all_sd_vals(3, 20);
  all_sd_vals.row(0) = sd_vals;
  all_sd_vals.row(1) = sd_r_vals;
  all_sd_vals.row(2) = sd_diff_vals;
}
