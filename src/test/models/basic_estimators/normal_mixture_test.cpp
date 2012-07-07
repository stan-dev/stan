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

  static size_t num_iterations() {
    return 8000U;
  }

  static void populate_chains() {
    if (chains->num_kept_samples() == 0U) {
      stan::mcmc::chains<> *tmp_chains = create_chains();
      for (size_t chain = 0U; chain < num_chains; chain++) {
        stan::mcmc::add_chain(*tmp_chains, chain, get_csv_file(chain), skip);
      }

      chains = create_chains();
      for (size_t chain = 0; chain < num_chains; chain++) {
	std::vector<double> theta, mu1, mu2, log_theta, log_one_minus_theta;
	tmp_chains->get_samples(chain, 0U, theta);
	tmp_chains->get_samples(chain, 1U, mu1);
	tmp_chains->get_samples(chain, 2U, mu2);
	tmp_chains->get_samples(chain, 3U, log_theta);
	tmp_chains->get_samples(chain, 4U, log_one_minus_theta);
	
	// if theta is > 0.5, swap values
	if (stan::math::mean(theta) > 0.5) {
	  for (size_t n = 0; n < theta.size(); n++) {
	    theta[n] = 1 - theta[n];
	  }
	  std::vector<double> tmp;
	  tmp = mu1;
	  mu1 = mu2;
	  mu2 = tmp;
	  
	  tmp = log_theta;
	  log_theta = log_one_minus_theta;
	  log_one_minus_theta = tmp;
	}
	
	std::vector<double> params(5, 0);
	for (size_t n = 0; n < theta.size(); n++) {
	  params[0] = theta[n];
	  params[1] = mu1[n];
	  params[2] = mu2[n];
	  params[3] = log_theta[n];
	  params[4] = log_one_minus_theta[n];
	  
	  chains->add(chain, params);
	}
      }
      delete(tmp_chains);
    }
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;

    expected_values.push_back(make_pair(0U, 0.2916));
    expected_values.push_back(make_pair(1U, -10.001));
    expected_values.push_back(make_pair(2U, 10.026));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicEstimators_NormalMixture,
			      Model_Test_Fixture,
			      Models_BasicEstimators_NormalMixture);
