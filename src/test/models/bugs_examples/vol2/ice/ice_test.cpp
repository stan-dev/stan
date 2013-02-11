#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Ice : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Ice> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("ice");
    model_path.push_back("ice");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static size_t num_iterations() {
    return 8000U;
  }

  static std::vector<size_t> skip_chains_test() {
    std::vector<size_t> params_to_skip;
    std::vector<size_t> dims;
    size_t index;
    dims.push_back(4);
    index = chains->get_total_param_index(chains->param_name_to_index("logRR"),
					  dims);
    params_to_skip.push_back(index);
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    size_t index;
    std::vector<size_t> dims;
    dims.push_back(0);

    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("logRR"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, -1.075));
    expected_values.push_back(make_pair(index + 1U, -0.7717));
    expected_values.push_back(make_pair(index + 2U, -0.4721));
    expected_values.push_back(make_pair(index + 3U, -0.2016));
    
    expected_values.push_back(make_pair(index + 5U, 0.1588));
    expected_values.push_back(make_pair(index + 6U, 0.319));
    expected_values.push_back(make_pair(index + 7U, 0.4829));
    expected_values.push_back(make_pair(index + 8U, 0.6512));
    expected_values.push_back(make_pair(index + 9U, 0.8466));
    expected_values.push_back(make_pair(index +10U, 1.059));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma"),
					  dims);
    expected_values.push_back(make_pair(index, 0.05286));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Ice,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol2_Ice);
