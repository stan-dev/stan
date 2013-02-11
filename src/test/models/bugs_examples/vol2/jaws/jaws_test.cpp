#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Jaws : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Jaws> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("jaws");
    model_path.push_back("jaws");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static size_t num_iterations() {
    return 500U;
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
    dims.push_back(0);

    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("Sigma"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 6.701));  // Sigma[1,1]
    expected_values.push_back(make_pair(index + 4U, 6.496));  // Sigma[1,2]
    expected_values.push_back(make_pair(index + 8U, 6.704));  // Sigma[1,3]
    expected_values.push_back(make_pair(index +12U, 5.836));  // Sigma[1,4]

    expected_values.push_back(make_pair(index + 1U, 6.496));  // Sigma[2,1]
    expected_values.push_back(make_pair(index + 5U, 6.813));  // Sigma[2,2]
    expected_values.push_back(make_pair(index + 9U, 6.469));  // Sigma[2,3]
    expected_values.push_back(make_pair(index +13U, 6.242));  // Sigma[2,4]

    expected_values.push_back(make_pair(index + 2U, 6.074));  // Sigma[3,1]
    expected_values.push_back(make_pair(index + 6U, 6.469));  // Sigma[3,2]
    expected_values.push_back(make_pair(index +10U, 7.312));  // Sigma[3,3]
    expected_values.push_back(make_pair(index +14U, 7.302));  // Sigma[3,4]

    expected_values.push_back(make_pair(index + 3U, 5.836));  // Sigma[4,1]
    expected_values.push_back(make_pair(index + 7U, 6.242));  // Sigma[4,2]
    expected_values.push_back(make_pair(index +11U, 7.302));  // Sigma[4,3]
    expected_values.push_back(make_pair(index +15U, 7.91));   // Sigma[4,4]

    dims.resize(1);
    index = chains->get_total_param_index(chains->param_name_to_index("beta0"),
					  dims);
    expected_values.push_back(make_pair(index, 33.56));
    index = chains->get_total_param_index(chains->param_name_to_index("beta1"),
					  dims);
    expected_values.push_back(make_pair(index, 1.885));

    index = chains->get_total_param_index(chains->param_name_to_index("mu"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 48.64));
    expected_values.push_back(make_pair(index + 1U, 49.58));
    expected_values.push_back(make_pair(index + 2U, 50.53));
    expected_values.push_back(make_pair(index + 3U, 51.47));
    
    return expected_values;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Jaws,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol2_Jaws);
