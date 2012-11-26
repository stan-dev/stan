#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Dogs : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Dogs> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("dogs");
    model_path.push_back("dogs");
    return model_path;
  }
  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static size_t num_iterations() {
    return 2000U;
  }

  static std::vector<size_t> skip_chains_test() {
    std::vector<size_t> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;
    
    expected_values.push_back(make_pair(0U, -0.2446));  // alpha
    expected_values.push_back(make_pair(2U, 0.7833));   // A

    expected_values.push_back(make_pair(1U, -0.07886)); // beta
    expected_values.push_back(make_pair(3U, 0.9242));   // B

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Dogs,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Dogs);
