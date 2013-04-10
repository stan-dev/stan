#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Hearts : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Hearts> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("hearts");
    model_path.push_back("hearts");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations() {
    return iterations;
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

    expected_values.push_back(make_pair(chains->index("alpha"), -0.4809));
    expected_values.push_back(make_pair(chains->index("beta"), 0.6427));
    expected_values.push_back(make_pair(chains->index("delta"), 0.3144));
    expected_values.push_back(make_pair(chains->index("theta"), 0.5717));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Hearts,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Hearts);
