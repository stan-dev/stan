#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Orange : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Orange> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("orange");
    model_path.push_back("orange");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations() {
    return 8000;
  }

  static std::vector<int> skip_chains_test() {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("mu[1]"), 5.257));
    expected_values.push_back(make_pair(chains->index("mu[2]"), 2.211));    
    expected_values.push_back(make_pair(chains->index("mu[3]"), -5.869));

    expected_values.push_back(make_pair(chains->index("sigma[1]"), 0.2332));
    expected_values.push_back(make_pair(chains->index("sigma[2]"), 0.1383));
    expected_values.push_back(make_pair(chains->index("sigma[3]"), 0.1012));

    expected_values.push_back(make_pair(chains->index("sigma_C"), 8.065));

    expected_values.push_back(make_pair(chains->index("theta[1,1]"), 5.079));  // theta[1,1]
    expected_values.push_back(make_pair(chains->index("theta[1,2]"), 2.134));  // theta[1,2]
    expected_values.push_back(make_pair(chains->index("theta[1,3]"), -5.851)); // theta[1,3]

    expected_values.push_back(make_pair(chains->index("theta[2,1]"), 5.395));  // theta[2,1]
    expected_values.push_back(make_pair(chains->index("theta[2,2]"), 2.207));  // theta[2,2]
    expected_values.push_back(make_pair(chains->index("theta[2,3]"), -5.825)); // theta[2,3]

    expected_values.push_back(make_pair(chains->index("theta[3,1]"), 5.079));  // theta[3,1]
    expected_values.push_back(make_pair(chains->index("theta[3,2]"), 2.187));  // theta[3,2]
    expected_values.push_back(make_pair(chains->index("theta[3,3]"), -5.908)); // theta[3,3]

    expected_values.push_back(make_pair(chains->index("theta[4,1]"), 5.441));  // theta[4,1]
    expected_values.push_back(make_pair(chains->index("theta[4,2]"), 2.269));  // theta[4,2]
    expected_values.push_back(make_pair(chains->index("theta[4,3]"), -5.816)); // theta[4,3]

    expected_values.push_back(make_pair(chains->index("theta[5,1]"), 5.291));  // theta[5,1]
    expected_values.push_back(make_pair(chains->index("theta[5,2]"), 2.299));  // theta[5,2]
    expected_values.push_back(make_pair(chains->index("theta[5,3]"), -5.907)); // theta[5,3]
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Orange,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Orange);
