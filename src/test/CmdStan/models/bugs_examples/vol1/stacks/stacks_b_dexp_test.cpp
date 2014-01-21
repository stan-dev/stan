#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Stacks_B_DoubleExponential : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Stacks_B_DoubleExponential> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("stacks");
    model_path.push_back("stacks_b_dexp");
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

    expected_values.push_back(make_pair(chains->index("b[1]"), 0.831));
    expected_values.push_back(make_pair(chains->index("b[2]"), 0.7545));
    expected_values.push_back(make_pair(chains->index("b[3]"), -0.1152));

    expected_values.push_back(make_pair(chains->index("b0"), -38.78));

    expected_values.push_back(make_pair(chains->index("sigma"), 3.492));

    expected_values.push_back(make_pair(chains->index("outlier_1"), 0.0453));
    expected_values.push_back(make_pair(chains->index("outlier_3"), 0.0578));
    expected_values.push_back(make_pair(chains->index("outlier_4"), 0.2929));
    expected_values.push_back(make_pair(chains->index("outlier_21"), 0.59));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Stacks_B_DoubleExponential,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Stacks_B_DoubleExponential);
