#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Leukfr : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Leukfr> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("leukfr");
    model_path.push_back("leukfr");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
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
    dims.push_back(0U);
    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					  dims);
    expected_values.push_back(make_pair(index, 1.607));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma"),
					  dims);
    expected_values.push_back(make_pair(index, 0.2415));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Leukfr,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Leukfr);
