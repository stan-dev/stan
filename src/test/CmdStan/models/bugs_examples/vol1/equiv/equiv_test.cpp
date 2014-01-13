#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Equiv : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Equiv> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("equiv");
    model_path.push_back("equiv");
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
    params_to_skip.push_back(chains->index("equiv"));
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;
    
    expected_values.push_back(make_pair(chains->index("equiv"), 0.9976));
    expected_values.push_back(make_pair(chains->index("mu"), 1.437));
    expected_values.push_back(make_pair(chains->index("phi"), -0.008338));
    expected_values.push_back(make_pair(chains->index("pi"), -0.1802));
    expected_values.push_back(make_pair(chains->index("sigma1"), 0.1106));
    expected_values.push_back(make_pair(chains->index("sigma2"), 0.1399));
    expected_values.push_back(make_pair(chains->index("theta"), 0.993));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Equiv,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Equiv);
