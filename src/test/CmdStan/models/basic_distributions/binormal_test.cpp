#include <test/CmdStan/models/model_test_fixture.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <gtest/gtest.h>

class Models_BasicDistributions_Binormal : 
  public Model_Test_Fixture<Models_BasicDistributions_Binormal> {
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("binormal");
    return model_path;
  }
    
  static bool has_data() {
    return false;
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
    int index;
    
    // y1 should be 0.0
    index = chains->index("y[1]");
    expected_values.push_back(make_pair(index, 0.0));
    
    // y2 should be 0.0
    index = chains->index("y[2]");
    expected_values.push_back(make_pair(index, 0.0));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Binormal,
            Model_Test_Fixture,
            Models_BasicDistributions_Binormal);

