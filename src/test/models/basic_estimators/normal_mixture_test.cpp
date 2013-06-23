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
  static int num_iterations() {
    //return 8000;
    return 200;
  }

  static std::vector<int> skip_chains_test() {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static int find(const Eigen::Matrix<std::string,Eigen::Dynamic,1>& header,
                  const std::string& var) {
    for (int i = 0; i < header.size(); i++)
      if (header(i) == var)
        return i;
    return -1;
  }

  static void populate_chains() {
    if (chains->num_kept_samples() == 0) {
      for (int chain = 1; chain <= num_chains; chain++) {
        std::ifstream ifstream;
        stan::io::stan_csv stan_csv;
        ifstream.open(get_csv_file(chain).c_str());
        stan_csv = stan::io::stan_csv_reader::parse(ifstream);
        ifstream.close();
        
        int theta = find(stan_csv.header, "theta");
        int mu1 = find(stan_csv.header, "mu[1]");
        int mu2 = find(stan_csv.header, "mu[2]");
        int log_theta = find(stan_csv.header, "log_theta");
        int log_1mtheta = find(stan_csv.header, "log_one_minus_theta");

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
        chains->add(stan_csv);
      }
    }
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("theta"), 0.2916));
    expected_values.push_back(make_pair(chains->index("mu[1]"), -10.001));
    expected_values.push_back(make_pair(chains->index("mu[2]"), 10.026));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicEstimators_NormalMixture,
                              Model_Test_Fixture,
                              Models_BasicEstimators_NormalMixture);
