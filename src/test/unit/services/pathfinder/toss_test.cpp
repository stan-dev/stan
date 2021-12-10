#include <stan/services/pathfinder/single.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/eight_schools.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

auto&& blah = stan::math::init_threadpool_tbb(12);

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
  void operator()(const Eigen::VectorXd& vals) { eigen_states_.push_back(vals); }
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
  loggy logger;
  values parameter;
  values diagnostics;
  stan::io::array_var_context context;
  stan_model model;
};

stan::io::array_var_context init_init_context() {
  std::vector<std::string> names_r{"mu", "tau", "theta_tilde"};
  std::vector<double> values_r{0.516456,
  0.1732794, //-1.75285,
  -0.937965,
  -0.511504,
  0.291413,
  1.63283,
  -1.19327,
  1.59356,
  1.7787,
  0.643191};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_r{size_vec{}, size_vec{}, size_vec{8}};
  std::vector<std::string> names_i{""};
  std::vector<int> values_i{8};
  using size_vec = std::vector<size_t>;
  std::vector<size_vec> dims_i{size_vec{}};
  return stan::io::array_var_context(names_r, values_r, dims_r);
}

TEST_F(ServicesPathfinderSingle, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 2;
  double num_elbo_draws = 1000;
  double num_draws = 1000;
  int history_size = 15;
  double init_alpha = 0.001;
  double tol_obj = 0;
  double tol_rel_obj = 0;
  double tol_grad = 0;
  double tol_rel_grad = 0;
  double tol_param = 0;
  int num_iterations = 60;
  bool save_iterations = false;
  int refresh = 1;
  mock_callback callback;
  stan::io::array_var_context empty_context = init_init_context();


  Eigen::MatrixXd X_vals(21, 10);
  X_vals <<
  -0.9379653474316, -0.511504401452839, 0.291413453407586, 1.6328311599791, -1.19327227585018, 1.59355873987079, 1.77870107442141, 0.643191169947386, 0.516456175595522, -1.75285491812974, -0.667708, -0.363698, 0.208631, 1.1753, -0.857873, 1.14468, 1.28596, 0.463694, 0.631304, -1.45833, 0.150717, 0.0850697, -0.0371175, -0.151779, 0.117982, -0.168591, -0.116361, -0.0534748, 1.37109, 0.539936, 0.375857, 0.206779, -0.0804938, -0.18698, 0.143935, -0.271543, 0.00939887, -0.0479159, 1.81544, 1.61375, 0.878688, 0.437858, -0.1595, 0.100877, -0.110194, -0.209658, 0.841208, 0.124438, 2.56307, 2.87656, 0.835656, 0.28938, -0.157124, 0.175336, -0.169541, -0.094994, 0.746999, 0.177726, 2.5215, 2.76128, 0.862983, 0.222864, -0.166699, 0.224032, -0.20765, -0.0386151, 0.730413, 0.222437, 2.58218, 2.90326, 0.867223, 0.140358, -0.187308, 0.161857, -0.120468, -0.0464945, 0.479354, 0.275662, 2.64748, 3.18993, 0.76018, 0.200837, -0.165319, 0.145555, -0.133769, -0.0514939, 0.54871, 0.259796, 2.61469, 3.17284, 0.73976, 0.188719, -0.161185, 0.144243, -0.126722, -0.0540425, 0.532005, 0.259251, 2.61664, 3.23775, 0.715526, 0.17578, -0.152404, 0.138478, -0.121947, -0.053399, 0.497343, 0.23986, 2.59348, 3.28784, 0.718124, 0.174263, -0.151608, 0.141961, -0.116709, -0.048173, 0.502842, 0.236043, 2.57425, 3.29454, 0.721043, 0.177217, -0.150325, 0.143547, -0.11356, -0.0448803, 0.50873, 0.230468, 2.53103, 3.29763, 0.728927, 0.183963, -0.145169, 0.152236, -0.105835, -0.0330273, 0.523303, 0.219593, 2.33846, 3.30414, 0.738022, 0.197333, -0.136582, 0.160657, -0.0892523, -0.0232173, 0.534787, 0.223114, 2.00287, 3.31777, 0.742692, 0.204078, -0.123187, 0.169811, -0.092551, -0.00725547, 0.537874, 0.254255, 1.54852, 3.34801, 0.726131, 0.208904, -0.113476, 0.17482, -0.0608786, -0.00913573, 0.520924, 0.266993, 1.30314, 3.36102, 0.726111, 0.203815, -0.116663, 0.169299, -0.075497, -0.0127053, 0.513807, 0.262536, 1.42165, 3.36352, 0.723005, 0.202541, -0.116838, 0.168001, -0.0766521, -0.0128625, 0.510969, 0.262904, 1.43252, 3.36655, 0.723162, 0.202521, -0.117191, 0.167857, -0.0765819, -0.0131429, 0.510897, 0.263148, 1.43293, 3.36647, 0.723114, 0.202517, -0.117238, 0.167945, -0.0765771, -0.0130584, 0.510919, 0.263161, 1.43295, 3.3664;
  //std::cout << "X: \n" << X_vals << "\n";
  X_vals.transposeInPlace();
  /*
  std::cout << "X: \n" << X_vals << "\n";
  */
  Eigen::MatrixXd G_vals(21, 10);
  G_vals <<
  -0.959256330532746, -0.524625358486603, 0.293827813622171, 1.62395155240062, -1.19047053791481, 1.59326171139352, 1.74893990789051, 0.637109257575491, -0.407643819107661, -1.0453791562244, -0.696164, -0.381036, 0.211975, 1.16359, -0.853761, 1.14448, 1.24625, 0.455609, -0.396231, -1.04201, -0.0503883, -0.0261708, -0.00824623, -0.235295, 0.172499, -0.16743, -0.405122, -0.110251, -0.32796, -0.774569, -0.166413, -0.0516421, 0.00603523, -0.44111, 0.363287, -0.294291, -0.800955, -0.209493, -0.274606, -0.174514, 0.102485, 0.852647, 0.0299219, -0.287352, 0.241974, -0.526429, 0.751939, -0.271598, 0.0456244, 0.18645, -0.026246, 0.146924, 0.0304792, -0.0475403, -0.00561046, -0.0925512, 0.167844, -0.1478, -0.0115247, -0.530177, 0.0783262, -0.0240568, 0.0144019, 0.173865, -0.253575, 0.093704, 0.347543, -0.0793166, 0.0210828, -0.255826, 0.404111, -0.331698, -0.0831051, 0.0772494, 0.0959209, 0.0575332, -0.421812, 0.0764533, 0.0360866, -0.100208, -0.00763214, 0.0599035, -0.00978904, -0.0340339, -0.00969479, 0.0245239, 0.00321709, 0.0252715, 0.0412275, -0.12862, -0.00041446, 0.0421016, -0.0108922, -0.00494786, -0.00461425, -0.00354606, 0.065832, 0.0407688, 0.0498011, -0.0179956, -0.0274452, -0.0112481, 0.00572433, -0.0158935, -0.0137745, -0.0172758, -0.0611676, -0.0066508, 0.044491, -0.0293329, -0.00832126, -0.0217078, 0.00493035, 0.00874475, 0.0255073, 0.0131718, -0.000497742, -0.0186966, 0.0490626, 0.00226387, 0.00385592, -0.00549597, 0.00446075, 0.0124978, 0.0398475, 0.0260008, 0.0465842, -0.0396117, 0.0500701, 0.02438, 0.0251435, 0.00615212, 0.00225265, 0.035934, 0.0477928, 0.0658124, 0.138207, -0.0898899, 0.0460609, 0.0609148, 0.0476127, 0.0452688, -0.00361461, 0.0322037, 0.0946054, 0.0593727, 0.193221, -0.103936, 0.0351806, 0.0841389, 0.0694903, 0.0202448, -0.00714195, 0.0238126, -0.122116, 0.0731776, 0.21044, -0.0283565, 0.0111021, 0.117655, -0.0130051, 0.0139338, 0.00280393, 0.0179159, 0.134343, 0.000358796, 0.0354597, -6.73221e-05, -0.000705955, -0.0126725, 0.00699117, 0.0044783, 0.00196504, 0.0052278, 0.0102796, 0.000328956, 0.0128139, -0.00438954, 0.000342617, 0.00330117, -0.000297419, 0.00022161, 0.00165241, 0.000571008, -0.00106536, 0.00145726, 0.000811386, -0.00084998, 6.6186e-05, 0.00028314, 0.000359043, 4.55642e-05, 0.000207156, -0.000543046, -7.04233e-05, -0.000666129, -1.12489e-05, 3.26843e-05, -1.65258e-05, 0.000152705, -1.34697e-05, -0.000103397, 2.50313e-05, 8.99411e-05, 4.38511e-05, 1.53115e-05, -8.10758e-05, 5.07464e-05, -4.18918e-07, -6.97718e-05;
 //std::cout << "G: \n" << G_vals << "\n";
  G_vals.transposeInPlace();

  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;
  for (Eigen::Index i = 0; i < X_vals.cols(); ++i) {
      input_iters.emplace_back(X_vals.col(i), G_vals.col(i));
  }

  int return_code = stan::services::optimize::pathfinder_lbfgs_single(//X_vals, G_vals,
      model, empty_context, seed, chain, init_radius, history_size, init_alpha,
      tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param, num_iterations, save_iterations, refresh, callback, num_elbo_draws, num_draws, 1,
      logger, init, parameter, diagnostics);



       //Eigen::MatrixXd param_vals = parameter.values_.transpose();
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
       std::cout << "Mean Values: \n" << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
       std::cout << "SD Values: \n"
                 << (((param_vals.colwise() - mean_vals)
                         .array()
                         .square()
                         .matrix()
                         .rowwise()
                         .sum()
                         .array()
                     / (param_vals.cols() - 1))
                        .sqrt()).transpose().eval()
                 << "\n";

}
