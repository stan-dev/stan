#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Seeds : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Seeds> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("seeds");
    model_path.push_back("seeds");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations() {
    return 20000;
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

    expected_values.push_back(make_pair(chains->index("alpha0"), -0.5525));  // alpha0
    expected_values.push_back(make_pair(chains->index("alpha1"), 0.08382));  // alpha1
    expected_values.push_back(make_pair(chains->index("alpha12"), -0.8165));  // alpha12
    expected_values.push_back(make_pair(chains->index("alpha2"), 1.346));    // alpha2
    expected_values.push_back(make_pair(chains->index("sigma"), 0.267));   // sigma

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Seeds,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Seeds);
