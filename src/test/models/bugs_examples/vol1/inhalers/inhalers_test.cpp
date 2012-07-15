#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Inhalers : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Inhalers> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("inhalers");
    model_path.push_back("inhalers");
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

    index = chains->get_total_param_index(chains->param_name_to_index("a"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 0.712));
    expected_values.push_back(make_pair(index + 1U, 3.936));
    expected_values.push_back(make_pair(index + 2U, 5.28));

    index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					  dims);
    expected_values.push_back(make_pair(index, 1.067));

    index = chains->get_total_param_index(chains->param_name_to_index("kappa"),
					  dims);
    expected_values.push_back(make_pair(index, 0.2463));

    index = chains->get_total_param_index(chains->param_name_to_index("log_sigma"),
					  dims);
    expected_values.push_back(make_pair(index, 0.195));

    index = chains->get_total_param_index(chains->param_name_to_index("pi"),
					  dims);
    expected_values.push_back(make_pair(index, -0.2367));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma"),
					  dims);
    expected_values.push_back(make_pair(index, 1.24));




    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Inhalers,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Inhalers);
