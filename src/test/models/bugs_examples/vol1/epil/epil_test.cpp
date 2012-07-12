#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Epil : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Epil> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("epil");
    model_path.push_back("epil");
    return model_path;
  }
  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static size_t num_iterations() {
    return iterations;
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
    
    index = chains->get_total_param_index(chains->param_name_to_index("alpha_Age"),
					  dims);
    expected_values.push_back(make_pair(index, 0.4891));

    index = chains->get_total_param_index(chains->param_name_to_index("alpha_BT"),
					  dims);
    expected_values.push_back(make_pair(index, 0.3496));

    index = chains->get_total_param_index(chains->param_name_to_index("alpha_Base"),
					  dims);
    expected_values.push_back(make_pair(index, 0.8931));

    index = chains->get_total_param_index(chains->param_name_to_index("alpha_Trt"),
					  dims);
    expected_values.push_back(make_pair(index, -0.9428));

    index = chains->get_total_param_index(chains->param_name_to_index("alpha_V4"),
					  dims);
    expected_values.push_back(make_pair(index, -0.1027));

    index = chains->get_total_param_index(chains->param_name_to_index("alpha0"),
					  dims);
    expected_values.push_back(make_pair(index, -1.435));
    
    index = chains->get_total_param_index(chains->param_name_to_index("sigma_b"),
					  dims);
    expected_values.push_back(make_pair(index, 0.3624));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma_b1"),
					  dims);
    expected_values.push_back(make_pair(index, 0.4987));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Epil,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Epil);
