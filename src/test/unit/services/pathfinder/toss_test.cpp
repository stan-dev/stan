#include <stan/services/pathfinder/single.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
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

stan::io::array_var_context init_init_context() {
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
}

TEST_F(ServicesPathfinderSingle, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 2;

  bool save_iterations = true;
  int refresh = 0;
  mock_callback callback;
  stan::io::array_var_context empty_context = init_init_context();


  Eigen::MatrixXd X_vals(21, 10);
  X_vals << -0.9379653474316, -0.511504401452839, 0.291413453407586, 1.6328311599791, -1.19327227585018, 1.59355873987079, 1.77870107442141, 0.643191169947386, 0.516456175595522, -1.75285491812974, -0.667708, -0.363698, 0.208631, 1.1753, -0.857873, 1.14468, 1.28596, 0.463694, 0.631304, -1.45833, 0.150717, 0.0850697, -0.0371175, -0.151779, 0.117982, -0.168591, -0.116361, -0.0534748, 1.37109, 0.539936, 0.375857, 0.206779, -0.0804938, -0.18698, 0.143935, -0.271543, 0.00939887, -0.0479159, 1.81544, 1.61375, 0.878688, 0.437858, -0.1595, 0.100877, -0.110194, -0.209658, 0.841208, 0.124438, 2.56307, 2.87656, 0.835656, 0.28938, -0.157124, 0.175336, -0.169541, -0.094994, 0.746999, 0.177726, 2.5215, 2.76128, 0.862983, 0.222864, -0.166699, 0.224032, -0.20765, -0.0386151, 0.730413, 0.222437, 2.58218, 2.90326, 0.867223, 0.140358, -0.187308, 0.161857, -0.120468, -0.0464945, 0.479354, 0.275662, 2.64748, 3.18993, 0.760522, 0.200862, -0.16539, 0.145554, -0.133813, -0.0515679, 0.548793, 0.259809, 2.61472, 3.17261, 0.744825, 0.189933, -0.16231, 0.14342, -0.128625, -0.0561653, 0.534602, 0.258158, 2.61826, 3.23213, 0.717211, 0.175934, -0.153722, 0.140003, -0.119349, -0.0529838, 0.500412, 0.24269, 2.5998, 3.28472, 0.719301, 0.175416, -0.152313, 0.141059, -0.120398, -0.0487121, 0.503313, 0.237734, 2.57899, 3.29236, 0.724619, 0.180577, -0.148156, 0.139524, -0.116703, -0.0438697, 0.508983, 0.222961, 2.49285, 3.3053, 0.729881, 0.183836, -0.142353, 0.146174, -0.11508, -0.0368745, 0.515395, 0.212486, 2.32992, 3.31499, 0.739335, 0.198301, -0.128901, 0.157075, -0.0883981, -0.0169649, 0.52648, 0.216963, 1.8753, 3.34281, 0.739108, 0.202998, -0.121663, 0.160346, -0.099496, -0.0176127, 0.522836, 0.231371, 1.60923, 3.35561, 0.730678, 0.202588, -0.116716, 0.168322, -0.0800608, -0.0116384, 0.516369, 0.250822, 1.42912, 3.36723, 0.723586, 0.203329, -0.11686, 0.168272, -0.0757917, -0.0125344, 0.51102, 0.262395, 1.41759, 3.36748, 0.723036, 0.2024, -0.117266, 0.16799, -0.0764956, -0.0130282, 0.510947, 0.263088, 1.43139, 3.36654, 0.723066, 0.202636, -0.117258, 0.167919, -0.0766326, -0.0130649, 0.510858, 0.263196, 1.43384, 3.36644, 0.723082, 0.202519, -0.11725, 0.167922, -0.0765784, -0.0130602, 0.5109, 0.263129, 1.43295, 3.36646;
  X_vals.transposeInPlace();
  /*
  std::cout << "X: \n" << X_vals << "\n";
  */
  Eigen::MatrixXd G_vals(21, 10);
  G_vals <<
  -0.959256330532746,
  -0.524625358486603,
  0.293827813622171,
  1.62395155240062,
  -1.19047053791481,
   1.59326171139352,
   1.74893990789051,
   0.637109257575491,
   -0.407643819107661,
    -1.0453791562244,

     -0.696164, -0.381036, 0.211975, 1.16359, -0.853761, 1.14448, 1.24625, 0.455609, -0.396231, -1.04201, -0.0503883, -0.0261708, -0.00824623, -0.235295, 0.172499, -0.16743, -0.405122, -0.110251, -0.32796, -0.774569, -0.166413, -0.0516421, 0.00603523, -0.44111, 0.363287, -0.294291, -0.800955, -0.209493, -0.274606, -0.174514, 0.102485, 0.852647, 0.0299219, -0.287352, 0.241974, -0.526429, 0.751939, -0.271598, 0.0456244, 0.18645, -0.026246, 0.146924, 0.0304792, -0.0475403, -0.00561046, -0.0925512, 0.167844, -0.1478, -0.0115247, -0.530177, 0.0783262, -0.0240568, 0.0144019, 0.173865, -0.253575, 0.093704, 0.347543, -0.0793166, 0.0210828, -0.255826, 0.404111, -0.331698, -0.0831051, 0.0772494, 0.0959209, 0.0575332, -0.421812, 0.0764533, 0.0360866, -0.100208, -0.00668878, 0.0598439, -0.00996546, -0.0341485, -0.00985414, 0.0241465, 0.003184, 0.0252585, 0.0412031, -0.128593, 0.0115433, 0.0455216, -0.0132406, -0.0132699, -0.0161169, -0.015258, 0.0688896, 0.035968, 0.0486862, -0.018348, -0.0244444, -0.0116358, 0.00223098, -0.00605959, 0.0167981, -0.012148, -0.043872, 0.00204626, 0.0467939, -0.0262796, -0.00624339, -0.0132504, 0.00332481, 0.00194178, -0.00761448, 0.0109535, -0.00217212, -0.0134294, 0.0480837, 0.00431829, 0.0272839, 0.021067, 0.00671878, -0.0190974, -0.0117472, 0.0230575, 0.0636285, -0.0655847, 0.0475213, 0.0495106, 0.0470598, 0.0158007, 0.00945582, -0.00100609, -0.0598012, 0.0347876, 0.106595, -0.112175, 0.040412, 0.0750226, 0.0849901, 0.0531053, 0.00679999, -0.0018934, 0.0421853, 0.0754665, 0.179501, -0.131089, 0.0292561, 0.113115, 0.0759473, 0.0389802, 0.00396353, -0.0279177, -0.185352, 0.00711786, 0.120206, -0.101169, 0.00554397, 0.0926815, 0.0369682, 0.00070456, 0.00160514, 0.00299219, -0.0415748, 0.0103198, 0.0532285, -0.0443057, 0.000594561, 0.036326, 0.00237277, 0.00470545, -0.000380505, 7.87629e-05, 0.00271171, 0.00041567, 0.000491839, -0.00362602, -0.000411932, 0.00131824, -0.00030719, -0.00144696, -0.000306564, 0.000210834, 0.000310076, -0.000130793, 0.00025329, -0.000293187, -0.000111377, -0.000257, -5.81535e-05, 0.00132304, 3.18412e-05, 0.000125559, -0.000295044, 0.000173346, -0.000266433, 0.000269009, 8.31465e-05, 0.000242928, -4.36696e-05, 9.06244e-06, -3.96597e-05, -3.86122e-05, -1.48687e-05, -4.21378e-06, -3.61292e-05, -3.96075e-05, -1.23072e-06, 1.31628e-05;
  G_vals.transposeInPlace();
/*
  std::cout << "G: \n" << G_vals << "\n";
*/
  std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> input_iters;
  for (Eigen::Index i = 0; i < X_vals.cols(); ++i) {
      input_iters.emplace_back(X_vals.col(i), G_vals.col(i));
  }

  int return_code = stan::services::optimize::pathfinder_lbfgs_single(
      model, input_iters, empty_context, seed, chain, init_radius, 6, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, callback, 100, 100, 1,
      logger, init, parameter, diagnostics);



       Eigen::MatrixXd param_vals = parameter.values_.transpose();

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
        std::cout << "Values: \n"
                  << param_vals.format(CommaInitFmt) << "\n";
         Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
         std::cout << "Mean Values: \n"
                   << mean_vals.format(CommaInitFmt) << "\n";
         std::cout << "SD Values: \n" << ((param_vals.rowwise() - mean_vals).array().square().matrix().colwise().sum().array() / (param_vals.rows() - 1)).sqrt() << "\n";


}
