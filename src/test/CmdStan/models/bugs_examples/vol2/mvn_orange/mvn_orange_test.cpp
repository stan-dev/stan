// FIXME v2.1.0: the parameters of this model doesn't always converge
#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_MvnOrange : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_MvnOrange> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("mvn_orange");
    model_path.push_back("mvn_orange");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations() {
    return 10000;
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

    expected_values.push_back(make_pair(chains->index("mu[1]"), 5.265));
    expected_values.push_back(make_pair(chains->index("mu[2]"), 2.2));    
    expected_values.push_back(make_pair(chains->index("mu[3]"), -5.88));

    expected_values.push_back(make_pair(chains->index("sigma[1]"), 0.2581));
    expected_values.push_back(make_pair(chains->index("sigma[2]"), 0.2679));
    expected_values.push_back(make_pair(chains->index("sigma[3]"), 0.2296));

    expected_values.push_back(make_pair(chains->index("sigma_C"), 7.853));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(DISABLED_Models_BugsExamples_Vol2_MvnOrange,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_MvnOrange);
