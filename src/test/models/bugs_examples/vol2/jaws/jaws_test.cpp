#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Jaws : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Jaws> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("jaws");
    model_path.push_back("jaws");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations() {
    return 500;
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

    expected_values.push_back(make_pair(chains->index("Sigma[1,1]"), 6.701));  // Sigma[1,1]
    expected_values.push_back(make_pair(chains->index("Sigma[1,2]"), 6.496));  // Sigma[1,2]
    expected_values.push_back(make_pair(chains->index("Sigma[1,3]"), 6.704));  // Sigma[1,3]
    expected_values.push_back(make_pair(chains->index("Sigma[1,4]"), 5.836));  // Sigma[1,4]

    expected_values.push_back(make_pair(chains->index("Sigma[2,1]"), 6.496));  // Sigma[2,1]
    expected_values.push_back(make_pair(chains->index("Sigma[2,2]"), 6.813));  // Sigma[2,2]
    expected_values.push_back(make_pair(chains->index("Sigma[2,3]"), 6.469));  // Sigma[2,3]
    expected_values.push_back(make_pair(chains->index("Sigma[2,4]"), 6.242));  // Sigma[2,4]

    expected_values.push_back(make_pair(chains->index("Sigma[3,1]"), 6.074));  // Sigma[3,1]
    expected_values.push_back(make_pair(chains->index("Sigma[3,2]"), 6.469));  // Sigma[3,2]
    expected_values.push_back(make_pair(chains->index("Sigma[3,3]"), 7.312));  // Sigma[3,3]
    expected_values.push_back(make_pair(chains->index("Sigma[3,4]"), 7.302));  // Sigma[3,4]

    expected_values.push_back(make_pair(chains->index("Sigma[4,1]"), 5.836));  // Sigma[4,1]
    expected_values.push_back(make_pair(chains->index("Sigma[4,2]"), 6.242));  // Sigma[4,2]
    expected_values.push_back(make_pair(chains->index("Sigma[4,3]"), 7.302));  // Sigma[4,3]
    expected_values.push_back(make_pair(chains->index("Sigma[4,4]"), 7.91));   // Sigma[4,4]

    expected_values.push_back(make_pair(chains->index("beta0"), 33.56));
    expected_values.push_back(make_pair(chains->index("beta1"), 1.885));

    expected_values.push_back(make_pair(chains->index("mu[1]"), 48.64));
    expected_values.push_back(make_pair(chains->index("mu[2]"), 49.58));
    expected_values.push_back(make_pair(chains->index("mu[3]"), 50.53));
    expected_values.push_back(make_pair(chains->index("mu[4]"), 51.47));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Jaws,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Jaws);
