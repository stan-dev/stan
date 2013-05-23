#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicEstimators_NormalMixture : 
  public Model_Test_Fixture<Models_BasicEstimators_NormalMixture> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_estimators");
    model_path.push_back("normal_mixture");
    return model_path;
  }
  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
    
  }

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(200); //iterations for nuts
    num_iter.push_back(2000); //iterations for unit_metro
    num_iter.push_back(2000); //iterations for diag_metro
    num_iter.push_back(2000); //iterations for dense_metro
    return num_iter[i];
  }

  static std::vector<int> skip_chains_test(int i) {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains(int i) {
    if (chains[i]->num_kept_samples() == 0) {
      for (int chain = 0; chain < num_chains; chain++) {
  std::ifstream ifstream;
  stan::io::stan_csv stan_csv;
  ifstream.open(get_csv_file(chain).c_str());
  stan_csv = stan::io::stan_csv_reader::parse(ifstream);
  ifstream.close();


  int theta = 3;
  int mu1 = 4;
  int mu2 = 5;
  int log_theta = 6;
  int log_1mtheta = 7;
  // if theta > 0.5, swap values
  if (stan_csv.samples.col(theta).mean() > 0.5) {
    stan_csv.samples.col(theta) = 1.0 - stan_csv.samples.col(theta).array();
    
    Eigen::VectorXd tmp;
    tmp = stan_csv.samples.col(mu1);
    stan_csv.samples.col(mu1) = stan_csv.samples.col(mu2);
    stan_csv.samples.col(mu2) = tmp;
    
    tmp = stan_csv.samples.col(log_theta);
    stan_csv.samples.col(log_theta) = stan_csv.samples.col(log_1mtheta);
    stan_csv.samples.col(log_1mtheta) = tmp;
  }

  chains[i]->add(stan_csv);
      }
    }
  }

  static std::vector<std::pair<int, double> >
  get_expected_values(int i) {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains[i]->index("theta"), 0.2916));
    expected_values.push_back(make_pair(chains[i]->index("mu[1]"), -10.001));
    expected_values.push_back(make_pair(chains[i]->index("mu[2]"), 10.026));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicEstimators_NormalMixture,
            Model_Test_Fixture,
            Models_BasicEstimators_NormalMixture);
