#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Orange : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Orange> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("orange");
    model_path.push_back("orange");
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

    index = chains->get_total_param_index(chains->param_name_to_index("mu"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 5.257));
    expected_values.push_back(make_pair(index + 1U, 2.211));    
    expected_values.push_back(make_pair(index + 2U, -5.869));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 0.2332));
    expected_values.push_back(make_pair(index + 1U, 0.1383));
    expected_values.push_back(make_pair(index + 2U, 0.1012));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma_C"),
					  dims);
    expected_values.push_back(make_pair(index, 8.065));

    dims[0] = 0;
    dims.push_back(0);
    index = chains->get_total_param_index(chains->param_name_to_index("theta"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 5.079));  // theta[1,1]
    expected_values.push_back(make_pair(index + 1U, 5.395));  // theta[2,1]
    expected_values.push_back(make_pair(index + 2U, 5.079));  // theta[3,1]
    expected_values.push_back(make_pair(index + 3U, 5.441));  // theta[4,1]
    expected_values.push_back(make_pair(index + 4U, 5.291));  // theta[5,1]

    expected_values.push_back(make_pair(index + 5U, 2.134));  // theta[1,2]
    expected_values.push_back(make_pair(index + 6U, 2.207));  // theta[2,2]
    expected_values.push_back(make_pair(index + 7U, 2.187));  // theta[3,2]
    expected_values.push_back(make_pair(index + 8U, 2.269));  // theta[4,2]
    expected_values.push_back(make_pair(index + 9U, 2.299));  // theta[5,2]

    expected_values.push_back(make_pair(index +10U, -5.851)); // theta[1,3]
    expected_values.push_back(make_pair(index +11U, -5.825)); // theta[2,3]
    expected_values.push_back(make_pair(index +12U, -5.908)); // theta[3,3]
    expected_values.push_back(make_pair(index +13U, -5.816)); // theta[4,3]
    expected_values.push_back(make_pair(index +14U, -5.907)); // theta[5,3]
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Orange,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol2_Orange);
