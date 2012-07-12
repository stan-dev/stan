#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Mice : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Mice> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("mice");
    model_path.push_back("mice");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static size_t num_iterations() {
    return 4000U;
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
    dims.push_back(0U);

    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("median"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 23.65));
    /*expected_values.push_back(make_pair(index + 1U, 35.18));
    expected_values.push_back(make_pair(index + 2U, 26.68));
    expected_values.push_back(make_pair(index + 3U, 21.28));

    index = chains->get_total_param_index(chains->param_name_to_index("pos_control"),
					  dims);
    expected_values.push_back(make_pair(index, 0.3088));
    */

    index = chains->get_total_param_index(chains->param_name_to_index("r"),
					  dims);
    expected_values.push_back(make_pair(index, 2.902));

    /*index = chains->get_total_param_index(chains->param_name_to_index("test_sub"),
					  dims);
    expected_values.push_back(make_pair(index, -0.3475));
    
    index = chains->get_total_param_index(chains->param_name_to_index("veh_control"),
					  dims);
    expected_values.push_back(make_pair(index, -1.143));
    */

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Mice,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Mice);
