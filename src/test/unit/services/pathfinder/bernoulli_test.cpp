#include <stan/services/pathfinder/single.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/bernoulli.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

auto&& blah = stan::math::init_threadpool_tbb(4);

struct mock_callback : public stan::callbacks::interrupt {
  int n;
  mock_callback() : n(0) {}

  void operator()() { n++; }
};

class loggy : public stan::callbacks::logger {
  /**
   * Logs a message with debug log level
   *
   * @param[in] message message
   */
  virtual void debug(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with debug log level.
   *
   * @param[in] message message
   */
  virtual void debug(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs a message with info log level.
   *
   * @param[in] message message
   */
  virtual void info(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with info log level.
   *
   * @param[in] message message
   */
  virtual void info(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs a message with warn log level.
   *
   * @param[in] message message
   */
  virtual void warn(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs a message with warn log level.
   *
   * @param[in] message message
   */
  virtual void warn(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs an error with error log level.
   *
   * @param[in] message message
   */
  virtual void error(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs an error with error log level.
   *
   * @param[in] message message
   */
  virtual void error(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }

  /**
   * Logs an error with fatal log level.
   *
   * @param[in] message message
   */
  virtual void fatal(const std::string& message) {
    std::cout << message << "\n";
  }

  /**
   * Logs an error with fatal log level.
   *
   * @param[in] message message
   */
  virtual void fatal(const std::stringstream& message) {
    std::cout << message.str() << "\n";
  }
};

class values : public stan::callbacks::stream_writer {
 public:
  std::vector<std::string> names_;
  std::vector<std::vector<double>> states_;
  std::vector<Eigen::VectorXd> eigen_states_;
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
  void operator()(
      const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>& xx) {
    optim_path_ = xx;
  }
  template <typename T, stan::require_eigen_vector_t<T>* = nullptr>
  void operator()(const T& vals) {
    eigen_states_.push_back(vals);
  }
  template <typename T, stan::require_eigen_dense_dynamic_t<T>* = nullptr>
  void operator()(const T& vals) {
    values_ = vals;
  }
};

constexpr size_t y_size = 10000;
stan::io::array_var_context init_context() {
  std::vector<std::string> names_r{"y"};
  boost::ecuyer1988 rng
      = stan::services::util::create_rng<boost::ecuyer1988>(123, 1);
  boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  std::vector<double> values_r(y_size);
  for (size_t i = 0; i < y_size; ++i) {
    values_r[i] = rand_unit_gaus() + 3;
  }
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{y_size}};
  std::vector<std::string> names_i{"N"};
  std::vector<int> values_i{y_size};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r, names_i,
                                     values_i, dims_i);
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
  loggy logger;
  values parameter;
  values diagnostics;
  stan::io::array_var_context context;
  stan_model model;
};

auto init_init_context() {
  std::vector<std::string> names_r{"theta"};
  std::vector<double> values_r{0};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{}};
  std::vector<std::string> names_i{};
  std::vector<int> values_i{};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{};
  return stan::io::array_var_context(names_r, values_r, dims_r);

  // return stan::io::empty_var_context();
}

TEST_F(ServicesPathfinderSingle, normal3_1) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 2;
  double num_elbo_draws = 10000;
  double num_draws = 10000;
  int history_size = 15;
  double init_alpha = 0.001;
  double tol_obj = 0;
  double tol_rel_obj = 0;
  double tol_grad = 0;
  double tol_rel_grad = 0;
  double tol_param = 0;
  int num_iterations = 2000;
  bool save_iterations = true;
  int refresh = 1;
  mock_callback callback;
  auto init_context = init_init_context();

  int return_code = stan::services::optimize::pathfinder_lbfgs_single(
      model, init_context, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
      save_iterations, refresh, callback, num_elbo_draws, num_draws, 1, logger,
      init, parameter, diagnostics);

  // Eigen::MatrixXd param_vals = parameter.values_.transpose();
  Eigen::MatrixXd param_vals = std::move(parameter.values_);

  std::cout << "\n --- Optim Path ---" << std::endl;
  for (Eigen::Index i = 0; i < diagnostics.optim_path_.size(); ++i) {
    Eigen::MatrixXd tmp(2, param_vals.cols() - 1);
    tmp.row(0) = std::get<0>(diagnostics.optim_path_[i]);
    tmp.row(1) = std::get<1>(diagnostics.optim_path_[i]);
    std::cout << "Iter: " << i << "\n" << tmp << "\n";
  }
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  std::cout << "---- Results  -------" << std::endl;
  /*
  std::cout << "Values: \n"
            << param_vals.format(CommaInitFmt) << "\n";
  */
  auto mean_vals = param_vals.rowwise().mean().eval();
  std::cout << "Mean Values: \n"
            << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
  std::cout << "SD Values: \n"
            << (((param_vals.colwise() - mean_vals)
                     .array()
                     .square()
                     .matrix()
                     .rowwise()
                     .sum()
                     .array()
                 / (param_vals.cols() - 1))
                    .sqrt())
                   .transpose()
                   .eval()
            << "\n";
}
