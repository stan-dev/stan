#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BasicEstimators_NormalLoc : 
  public Model_Test_Fixture<Models_BasicEstimators_NormalLoc> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_estimators");
    model_path.push_back("normal_loc");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
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

    expected_values.push_back(make_pair(chains->index("mu"), 1.15));
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicEstimators_NormalLoc,
            Model_Test_Fixture,
            Models_BasicEstimators_NormalLoc);
