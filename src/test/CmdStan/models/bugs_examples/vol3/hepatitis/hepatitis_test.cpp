#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol3_Hepatitis : 
  public Model_Test_Fixture<Models_BugsExamples_Vol3_Hepatitis> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol3");
    model_path.push_back("hepatitis");
    model_path.push_back("hepatitis");
    return model_path;
  }
  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations() {
    return 5000;
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

    expected_values.push_back(make_pair(chains->index("alpha0"), 6.138));
    expected_values.push_back(make_pair(chains->index("beta0"), -1.064));
    expected_values.push_back(make_pair(chains->index("gamma"), 0.6691)); 
    expected_values.push_back(make_pair(chains->index("sigma_y"), 1.003));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(DISABLED_Models_BugsExamples_Vol3_Hepatitis,
            Model_Test_Fixture,
            Models_BugsExamples_Vol3_Hepatitis);
