#include <stan/services/pathfinder/single.hpp>
#include <stan/io/array_var_context.hpp>
#include <test/test-models/good/stat_comp_benchmarks_models/eight_schools.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

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

class ServicesPathfinderSingle : public testing::Test {
 public:
  ServicesPathfinderSingle()
      : init(init_ss), context(init_context()), model(context, 0, &model_ss) {}

  std::stringstream init_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::io::array_var_context context;
  stan_model model;
};

TEST_F(ServicesPathfinderSingle, est_div) {
  Eigen::VectorXd alpha(10);
  alpha << 0.99434035015686, 0.996596991589148, 0.996912934085565,
      0.974233478220917, 0.985380229624499, 0.977804204019154,
      0.963518774743505, 0.993435151076477, 1.01061395944603, 1.09215336785842;
  Eigen::MatrixXd Qk(10, 2);
  Eigen::VectorXd Qk_vec(20);
  Qk_vec << -0.278474679167602, -0.152157033895653, 0.0867505466957011,
      0.482325877559475, -0.354786702113257, 0.471054430597708,
      0.523769553707243, 0.192024657691747, -0.012178549697562,
      -0.00373742904594976, -0.0101587357838484, -0.00545544981545159,
      -0.00195803909699599, -0.00223400842625933, 0.0120665390403349,
      0.00198288799603561, -0.00710942889371788, -0.0159165979874917,
      -0.34565269973477, -0.93804531304729;
  for (Eigen::Index i = 0; i < Qk_vec.size(); ++i) {
    Qk(i) = Qk_vec(i);
  }
  double logdetCholHk = 0.045802044684242;
  Eigen::VectorXd x_center(10);
  x_center << 0.148518589293209, 0.0837800375511616, -0.0363926200004713,
      -0.151859573239236, 0.116523128113017, -0.167778305381412,
      -0.119375573204488, -0.0521644831502672, 1.37484693636769,
      0.635081893690443;
  Eigen::MatrixXd chol(2, 2);
  chol << 1.01074004693027, 0, 0.311492962362723, 1.04641139117945;
  chol.transposeInPlace();
  Eigen::VectorXd mock_rando_vec(10 * 10);
  mock_rando_vec << -0.626453810742332, 0.183643324222082, -0.835628612410047,
      1.59528080213779, 0.329507771815361, -0.820468384118015,
      0.487429052428485, 0.738324705129217, 0.575781351653492,
      -0.305388387156356, 1.51178116845085, 0.389843236411431,
      -0.621240580541804, -2.2146998871775, 1.12493091814311,
      -0.0449336090152309, -0.0161902630989461, 0.943836210685299,
      0.821221195098089, 0.593901321217509, 0.918977371608218,
      0.782136300731067, 0.0745649833651906, -1.98935169586337,
      0.61982574789471, -0.0561287395290008, -0.155795506705329,
      -1.47075238389927, -0.47815005510862, 0.417941560199702, 1.35867955152904,
      -0.102787727342996, 0.387671611559369, -0.0538050405829051,
      -1.37705955682861, -0.41499456329968, -0.394289953710349,
      -0.0593133967111857, 1.10002537198388, 0.763175748457544,
      -0.164523596253587, -0.253361680136508, 0.696963375404737,
      0.556663198673657, -0.68875569454952, -0.70749515696212, 0.36458196213683,
      0.768532924515416, -0.112346212150228, 0.881107726454215,
      0.398105880367068, -0.612026393250771, 0.341119691424425,
      -1.12936309608079, 1.43302370170104, 1.98039989850586, -0.367221476466509,
      -1.04413462631653, 0.569719627442413, -0.135054603880824,
      2.40161776050478, -0.0392400027331692, 0.689739362450777,
      0.0280021587806661, -0.743273208882405, 0.188792299514343,
      -1.80495862889104, 1.46555486156289, 0.153253338211898, 2.17261167036215,
      0.475509528899663, -0.709946430921815, 0.610726353489055,
      -0.934097631644252, -1.2536334002391, 0.291446235517463,
      -0.443291873218433, 0.00110535163162413, 0.0743413241516641,
      -0.589520946188072, -0.568668732818502, -0.135178615123832,
      1.1780869965732, -1.52356680042976, 0.593946187628422, 0.332950371213518,
      1.06309983727636, -0.304183923634301, 0.370018809916288,
      0.267098790772231, -0.54252003099165, 1.20786780598317, 1.16040261569495,
      0.700213649514998, 1.58683345454085, 0.558486425565304, -1.27659220845804,
      -0.573265414236886, -1.22461261489836, -0.473400636439312;
  Eigen::MatrixXd mock_rando(10, 10);
  for (Eigen::Index i = 0; i < mock_rando_vec.size(); ++i) {
    mock_rando(i) = mock_rando_vec(i);
  }
  auto mock_rando_gen = [&mock_rando]() { return mock_rando; };
  auto my_mod = model;
  auto fn = [&my_mod](auto&& u) {
    return my_mod.template log_prob<false, false>(
        const_cast<std::decay_t<decltype(u)>&>(u), 0);
  };
  using stan::services::optimize::est_DIV;
  using stan::services::optimize::taylor_approx_t;
  taylor_approx_t tt{x_center, logdetCholHk, chol, Qk, false};
  auto xx = est_DIV(tt, 10, alpha, fn, mock_rando_gen);
  Eigen::MatrixXd param_vals = xx.repeat_draws;
  std::cout << "Params: \n" << param_vals << "\n";
  Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
  std::cout << "\n" << mean_vals << "\n";

  Eigen::RowVectorXd sd_vals = ((param_vals.rowwise() - mean_vals)
                                    .array()
                                    .square()
                                    .matrix()
                                    .colwise()
                                    .sum()
                                    .array()
                                / (param_vals.rows() - 1))
                                   .sqrt();
  std::cout << "\n" << sd_vals << "\n";
}
