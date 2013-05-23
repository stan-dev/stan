#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Pump : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Pump> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("pump");
    model_path.push_back("pump");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(2000); //iterations for nuts
    num_iter.push_back(5000); //iterations for unit_metro
    num_iter.push_back(5000); //iterations for diag_metro
    num_iter.push_back(5000); //iterations for dense_metro
    return num_iter[i];
  }

  static std::vector<int> skip_chains_test(int i) {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains(int i) {
    default_populate_chains(i);
  }

  static std::vector<std::pair<int, double> >
  get_expected_values(int i) {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;
    
    expected_values.push_back(make_pair(chains[i]->index("alpha"), 0.6867));   // alpha
    expected_values.push_back(make_pair(chains[i]->index("beta"), 0.905024)); // beta
    expected_values.push_back(make_pair(chains[i]->index("theta[1]"), 0.05986));  // theta[0]
    expected_values.push_back(make_pair(chains[i]->index("theta[2]"), 0.1015));   // theta[1]
    expected_values.push_back(make_pair(chains[i]->index("theta[3]"), 0.08899));  // theta[2]
    expected_values.push_back(make_pair(chains[i]->index("theta[4]"), 0.1156));   // theta[3]
    expected_values.push_back(make_pair(chains[i]->index("theta[5]"), 0.6043));   // theta[4]
    expected_values.push_back(make_pair(chains[i]->index("theta[6]"), 0.6121));   // theta[5]
    expected_values.push_back(make_pair(chains[i]->index("theta[7]"), 0.899));    // theta[6]
    expected_values.push_back(make_pair(chains[i]->index("theta[8]"), 0.9095));   // theta[7]
    expected_values.push_back(make_pair(chains[i]->index("theta[9]"), 1.587));    // theta[8]
    expected_values.push_back(make_pair(chains[i]->index("theta[10]"), 1.995));    // theta[9]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Pump,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Pump);
