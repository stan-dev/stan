// FIXME v2.1.0: the parameters of this model doesn't always converge
#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BasicDistributions_InvWishart : 
  public Model_Test_Fixture<Models_BasicDistributions_InvWishart> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("inv_wishart");
    return model_path;
  }

  static bool has_data() {
    return false;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations() {
    return 10000U;
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
    
    expected_values.push_back(make_pair(0U, 2));  // W[1,1]
    expected_values.push_back(make_pair(1U, 0));  // W[1,2]
    expected_values.push_back(make_pair(2U, 0));  // W[1,3]
    expected_values.push_back(make_pair(3U, 0));  // W[2,1]
    expected_values.push_back(make_pair(4U, 1));  // W[2,2]
    expected_values.push_back(make_pair(5U, 0));  // W[2,3]
    expected_values.push_back(make_pair(6U, 0));  // W[3,1]
    expected_values.push_back(make_pair(7U, 0));  // W[3,2]
    expected_values.push_back(make_pair(8U, 0.5));// W[3,3]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(DISABLED_Models_BasicDistributions_InvWishart,
            Model_Test_Fixture,
            Models_BasicDistributions_InvWishart);
