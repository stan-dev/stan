#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Salm2 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Salm2> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("salm");
    model_path.push_back("salm2");
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

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;

    size_t index;
    std::vector<size_t> dims;
    dims.push_back(0);
    
    index = chains->get_total_param_index(chains->param_name_to_index("alpha"),
					  dims);
    expected_values.push_back(make_pair(index, 2.193));
    
    index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					  dims);
    expected_values.push_back(make_pair(index, 0.3059));
    
    index = chains->get_total_param_index(chains->param_name_to_index("gamma"),
					  dims);
    expected_values.push_back(make_pair(index, -0.0009557));

    index = chains->get_total_param_index(chains->param_name_to_index("sigma"),
					  dims);
    expected_values.push_back(make_pair(index, 0.2608));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Salm2,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Salm2);
