#include <stan/math.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/eight_schools.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

auto&& blah = stan::math::init_threadpool_tbb(1);

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
  std::vector<std::vector<double> > states_;
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
  void operator()(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>& xx) {
    optim_path_ = xx;
  }
  template <typename EigVec, stan::require_eigen_vector_t<EigVec>* = nullptr>
  void operator()(const EigVec& vals) { eigen_states_.push_back(vals); }
  template <typename EigMat, stan::require_eigen_matrix_dynamic_t<EigMat>* = nullptr>
  void operator()(const EigMat& vals) { values_ = vals; }
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
  loggy logger;
  values parameter;
  values diagnostics;
  stan::io::array_var_context context;
  stan_model model;
};

auto init_init_context() {
  /*
  std::vector<std::string> names_r{"mu", "tau", "theta_tilde"};
  std::vector<double> values_r{0.3808,
  1.58,
   1.291413453407586,
    .16328311599791,
     -.119327227585018,
      .159355873987079,
       .177870107442141,
        .0643191169947386,
        .0516456175595522,
        -.0175285491812974};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{}, size_vec{}, size_vec{8}};
  std::vector<std::string> names_i{""};
  std::vector<int> values_i{8};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r);
  */
  return stan::io::empty_var_context();
}

TEST_F(ServicesPathfinderSingle, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 2;
  size_t num_multi_draws = 100;
  size_t num_threads = 1;
  size_t num_paths = 8;
  double num_elbo_draws = 1000;
  double num_draws = 100;
  int history_size = 15;
  double init_alpha = 0.00001;
  double tol_obj = 0;
  double tol_rel_obj = 0;
  double tol_grad = 0;
  double tol_rel_grad = 0;
  double tol_param = 0;
  int num_iterations = 60;
  //bool save_iterations = true;
  int refresh = 5;
  std::vector<stan::callbacks::writer> single_path_parameter_writer(num_paths);
  std::vector<stan::callbacks::writer> single_path_diagnostic_writer(num_paths);
  std::vector<std::unique_ptr<decltype(init_init_context())>> single_path_inits;
  for (int i = 0; i < num_paths; ++i) {
//    single_path_parameter_writer.emplace_back(single_path_parameter_ss[i]);
//    single_path_diagnostic_writer.emplace_back(single_path_diagnostic_ss[i]);
    single_path_inits.emplace_back(std::make_unique<decltype(init_init_context())>(init_init_context()));
  }
  bool save_iterations = false;
  //int refresh = 0;
  mock_callback callback;

  int return_code = stan::services::optimize::pathfinder_lbfgs_multi(
      model, single_path_inits, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations,
       save_iterations, refresh, callback, num_elbo_draws,
      num_draws, num_multi_draws, num_threads, num_paths,
      logger, std::vector<stan::callbacks::stream_writer>(num_paths, init),
       single_path_parameter_writer, single_path_diagnostic_writer, parameter, diagnostics);



       //Eigen::MatrixXd param_vals = parameter.values_.transpose();
       Eigen::MatrixXd param_vals(parameter.eigen_states_.size(), parameter.eigen_states_[0].size());
       for (size_t i = 0; i < parameter.eigen_states_.size(); ++i) {
         param_vals.row(i) = parameter.eigen_states_[i];
       }

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

        //std::cout << "Values: \n" << param_vals.format(CommaInitFmt) << "\n";

         Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
         std::cout << "Mean Values: \n"
                   << mean_vals.format(CommaInitFmt) << "\n";
         std::cout << "SD Values: \n" << ((param_vals.rowwise() - mean_vals).array().square().matrix().colwise().sum().array() / (param_vals.rows() - 1)).sqrt() << "\n";


}
