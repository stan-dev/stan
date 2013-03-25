#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Bones : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Bones> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("bones");
    model_path.push_back("bones");
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
    expected_values.push_back(make_pair(chains->index("theta[1]"), 0.3244));
    expected_values.push_back(make_pair(chains->index("theta[2]"), 1.366));
    expected_values.push_back(make_pair(chains->index("theta[3]"), 2.357));
    expected_values.push_back(make_pair(chains->index("theta[4]"), 2.902));
    expected_values.push_back(make_pair(chains->index("theta[5]"), 5.535));
    expected_values.push_back(make_pair(chains->index("theta[6]"), 6.751));
    expected_values.push_back(make_pair(chains->index("theta[7]"), 6.451));
    expected_values.push_back(make_pair(chains->index("theta[8]"), 8.93));
    expected_values.push_back(make_pair(chains->index("theta[9]"), 8.981));
    expected_values.push_back(make_pair(chains->index("theta[10]"), 11.94));
    expected_values.push_back(make_pair(chains->index("theta[11]"), 11.58));
    expected_values.push_back(make_pair(chains->index("theta[12]"), 15.79));
    expected_values.push_back(make_pair(chains->index("theta[13]"), 16.96));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Bones,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Bones);
