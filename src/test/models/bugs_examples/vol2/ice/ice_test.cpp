#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Ice : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Ice> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("ice");
    model_path.push_back("ice");
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
    params_to_skip.push_back(chains->index("logRR[5]"));
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("logRR[1]"), -1.075));
    expected_values.push_back(make_pair(chains->index("logRR[2]"), -0.7717));
    expected_values.push_back(make_pair(chains->index("logRR[3]"), -0.4721));
    expected_values.push_back(make_pair(chains->index("logRR[4]"), -0.2016));
    
    expected_values.push_back(make_pair(chains->index("logRR[6]"), 0.1588));
    expected_values.push_back(make_pair(chains->index("logRR[7]"), 0.319));
    expected_values.push_back(make_pair(chains->index("logRR[8]"), 0.4829));
    expected_values.push_back(make_pair(chains->index("logRR[9]"), 0.6512));
    expected_values.push_back(make_pair(chains->index("logRR[10]"), 0.8466));
    expected_values.push_back(make_pair(chains->index("logRR[11]"), 1.059));

    expected_values.push_back(make_pair(chains->index("sigma"), 0.05286));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Ice,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Ice);
