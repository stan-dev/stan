#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Magnesium : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Magnesium> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("magnesium");
    model_path.push_back("magnesium");
    return model_path;
  }

  static bool has_data() {
    return true;
  }
  
  static bool has_init() {
    return true;
  }

  static size_t num_iterations() {
    return 20000U;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    size_t OR_index,
      tau_index;
    std::vector<size_t> dims;
    dims.push_back(0);
    OR_index = chains->get_total_param_index(chains->param_name_to_index("OR"),
					     dims);
    tau_index = chains->get_total_param_index(chains->param_name_to_index("tau"),
					      dims);
    std::vector<std::pair<size_t, double> > expected_values;
    
    expected_values.push_back(make_pair(OR_index + 0U, 0.4591));
    expected_values.push_back(make_pair(tau_index + 0U, 0.5845));

    expected_values.push_back(make_pair(OR_index + 1U, 0.4179));
    expected_values.push_back(make_pair(tau_index + 1U, 1.081));
    
    expected_values.push_back(make_pair(OR_index + 2U, 0.4368));
    expected_values.push_back(make_pair(tau_index + 2U, 0.8119));

    expected_values.push_back(make_pair(OR_index + 3U, 0.4639));
    expected_values.push_back(make_pair(tau_index + 3U, 0.5084));

    expected_values.push_back(make_pair(OR_index + 4U, 0.483));
    expected_values.push_back(make_pair(tau_index + 4U, 0.5245));

    expected_values.push_back(make_pair(OR_index + 5U, 0.4347));
    expected_values.push_back(make_pair(tau_index + 5U, 0.5736));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Magnesium,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Magnesium);
