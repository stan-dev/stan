#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Kidney : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Kidney> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("kidney");
    model_path.push_back("kidney");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations() {
    return 20000;
  }

  static std::vector<int> skip_chains_test() {
    std::vector<int> params_to_skip;
    params_to_skip.push_back(chains->index("yabeta_disease[1]"));
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("alpha"), -5.529));

    expected_values.push_back(make_pair(chains->index("beta_disease2"), 0.1265));

    expected_values.push_back(make_pair(chains->index("beta_disease3"), 0.5995));

    expected_values.push_back(make_pair(chains->index("beta_disease4"), -1.198));

    expected_values.push_back(make_pair(chains->index("beta_sex"), -1.945));

    expected_values.push_back(make_pair(chains->index("r"), 1.205));

    expected_values.push_back(make_pair(chains->index("sigma"), 0.6367));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Kidney,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Kidney);
