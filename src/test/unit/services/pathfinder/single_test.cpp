#include <stan/services/pathfinder/single.hpp>
#include <stan/io/array_var_context.hpp>
#include <test/test-models/good/stat_comp_benchmarks_models/eight_schools.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

struct mock_callback : public stan::callbacks::interrupt {
  int n;
  mock_callback() : n(0) {}

  void operator()() { n++; }
};

class values : public stan::callbacks::stream_writer {
 public:
  std::vector<std::string> names_;
  std::vector<std::vector<double> > states_;
  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> optim_path_;
  Eigen::MatrixXd values_;
  values(std::ostream& stream) : stan::callbacks::stream_writer(stream) {}

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) { names_ = names; }

  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double>& state) {
    states_.push_back(state);
  }
  void operator()(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>& xx) {
    optim_path_ = xx;
  }
  void operator()(const Eigen::MatrixXd& vals) { values_ = vals; }
};

stan::io::array_var_context init_context() {
  std::vector<std::string> names_r{"y", "sigma"};
  std::vector<double> values_r{28, 8, -3, 7, -1, 1, 18, 12, 15, 10, 16, 11,  9, 11, 10, 18};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{8}, size_vec{8}};
  std::vector<std::string> names_i{"J"};
  std::vector<int> values_i{8};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r, names_i, values_i, dims_i);
}

class ServicesPathfinderSingle : public testing::Test {
 public:
  ServicesPathfinderSingle()
      : init(init_ss),
        parameter(parameter_ss),
        diagnostics(diagnostic_ss),
        context(init_context()),
        model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::unit::instrumented_logger logger;
  values parameter;
  values diagnostics;
  stan::io::array_var_context context;
  stan_model model;
};

TEST_F(ServicesPathfinderSingle, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  bool save_iterations = true;
  int refresh = 0;
  mock_callback callback;

  int return_code = stan::services::optimize::pathfinder_lbfgs_single(
      model, context, seed, chain, init_radius, 6, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, callback, 500, 100, 1,
      logger, init, parameter, diagnostics);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");
  double known_mu_mean{4.38};
  double known_tau_mean{0.12367541 };
  Eigen::VectorXd known_theta_tilde_mean(8);
  known_theta_tilde_mean << -0.03963756, -0.12260122, 0.06966011, -0.15862005, -0.12785432, -0.07130486,
    1.36489658, 0.62058815;
  double known_mu_sd{0.9958823};
  double known_tau_sd{0.9749844};
  Eigen::VectorXd known_theta_tilde_sd(8);
   known_theta_tilde_sd << 1.0034803, 0.9962264, 0.9927524, 0.9934484, 1.0353772, 1.0122802, 0.9880471, 1.1189115;

 Eigen::MatrixXd param_vals = parameter.values_.transpose();
 std::cout << "Values: \n"
           << param_vals.format(CommaInitFmt) << "\n";
  Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
  std::cout << "Mean Values: \n"
            << mean_vals.format(CommaInitFmt) << "\n";
  std::cout << "SD Values: \n" << ((param_vals.rowwise() - mean_vals).array().square().matrix().colwise().sum().array() / (param_vals.rows() - 1)).sqrt() << "\n";

  std::cout << "\n --- Optim Path ---" << std::endl;
  for (Eigen::Index i = 0; i < diagnostics.optim_path_.size(); ++i) {
    Eigen::MatrixXd tmp(2, param_vals.cols() - 1);
    tmp.row(0) = std::get<0>(diagnostics.optim_path_[i]);
    tmp.row(1) = std::get<1>(diagnostics.optim_path_[i]);
    std::cout << "Iter: " << i << "\n" << tmp << "\n";
  }
//            (().array()).square().sum()/(param_vals.rows() - 1)).sqrt();
  /*
    EXPECT_EQ(logger.call_count(), logger.call_count_info())
        << "all output to info";
    EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
    EXPECT_EQ(1, logger.find("Optimization terminated normally: "));
    EXPECT_EQ(1, logger.find("  Convergence detected: relative gradient "
                             "magnitude is below tolerance"));

    EXPECT_EQ("0,0\n", init_ss.str());

    ASSERT_EQ(3, parameter.names_.size());
    EXPECT_EQ("lp__", parameter.names_[0]);
    EXPECT_EQ("x", parameter.names_[1]);
    EXPECT_EQ("y", parameter.names_[2]);

    EXPECT_EQ(23, parameter.states_.size());
    EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
        << "initial value should be (0, 0)";
    EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
        << "initial value should be (0, 0)";
    EXPECT_FLOAT_EQ(0.99998301, parameter.states_.back()[1])
        << "optimal value should be (1, 1)";
    EXPECT_FLOAT_EQ(0.99996597, parameter.states_.back()[2])
        << "optimal value should be (1, 1)";
    EXPECT_FLOAT_EQ(return_code, 0);
    EXPECT_EQ(22, callback.n);
  */
}
