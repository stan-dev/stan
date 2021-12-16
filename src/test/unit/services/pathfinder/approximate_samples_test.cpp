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

TEST_F(ServicesPathfinderSingle, construct_taylor_approximation_sparse) {
  Eigen::VectorXd alpha(10);
  alpha << 0.99434035015686, 0.996596991589148, 0.996912934085565,
      0.974233478220917, 0.985380229624499, 0.977804204019154,
      0.963518774743505, 0.993435151076477, 1.01061395944603, 1.09215336785842;
  Eigen::VectorXd Ykt_h(10);
  Ykt_h << 0.263092330532746, 0.143589358486603, -0.0818528136221707,
      -0.460361552400624, 0.33670953791481, -0.448781711393516,
      -0.502689907890507, -0.181500257575491, 0.0114128191076615,
      0.00336915622440426;
  Eigen::VectorXd Skt_h(10);
  Skt_h << 0.2702573474316, 0.147806401452839, -0.0827824534075856,
      -0.457531159979105, 0.335399275850177, -0.448878739870787,
      -0.492741074421406, -0.179497169947386, 0.114847824404478,
      0.294524918129742;
  const auto param_size = Ykt_h.rows();
  const auto m = Ykt_h.cols();
  Eigen::VectorXd Dk(m);
  Dk[0] = Ykt_h.dot(Skt_h);
  std::cout << "Dk: \n" << Dk << "\n";
  Eigen::MatrixXd Rk = Eigen::MatrixXd::Zero(m, m);
  Rk(0) = Skt_h.dot(Ykt_h);
  Eigen::MatrixXd ninvRST;
  {
    Eigen::MatrixXd Skt_mat(param_size, m);
    for (Eigen::Index i = 0; i < m; ++i) {
      Skt_mat.col(i) = Skt_h;
    }
    ninvRST = -(Rk.triangularView<Eigen::Upper>().solve(Skt_mat.transpose()));
  }
  std::cout << "ninvRST: \n" << ninvRST << "\n";
  Eigen::MatrixXd Ykt_mat(param_size, m);
  for (Eigen::Index i = 0; i < m; ++i) {
    Ykt_mat.col(i) = Ykt_h;
  }
  bool ill_dist = false;
  Eigen::VectorXd point_est(10);
  point_est << -0.667708, -0.363698, 0.208631, 1.1753, -0.857873, 1.14468,
      1.28596, 0.463694, 0.631304, -1.45833;
  Eigen::VectorXd grad_est(10);
  grad_est << -0.696164, -0.381036, 0.211975, 1.16359, -0.853761, 1.14448,
      1.24625, 0.455609, -0.396231, -1.04201;
  using stan::services::optimize::construct_taylor_approximation_sparse;
  auto xx = construct_taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                  point_est, grad_est);

  std::cout << "wkbart: \n" << xx.wkbart << "\n";
  std::cout << "mkbart: \n" << xx.mkbart << "\n";
  std::cout << "L_approx: \n" << xx.L_approx << "\n";
  std::cout << "Rkbar: \n" << xx.Rkbar << "\n";
  std::cout << "x_center: \n" << xx.x_center << "\n";
  std::cout << "Qk: \n" << xx.Qk << "\n";
  std::cout << "logdet" << xx.logdetcholHk << "\n";
  /*
Eigen::MatrixXd param_vals = xx.repeat_draws;
std::cout << "Params: \n" << param_vals << "\n";
Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
std::cout << "\n" << mean_vals << "\n";

Eigen::RowVectorXd sd_vals = ((param_vals.rowwise() -
mean_vals).array().square().matrix().colwise().sum().array() /
(param_vals.rows() - 1)).sqrt(); std::cout << "\n" << sd_vals << "\n";
*/
}
