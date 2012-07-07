#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Rats : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Rats> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("rats");
    model_path.push_back("rats");
    return model_path;
  }

  static bool has_data() {
    return true;
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
    size_t index;
    std::vector<size_t> idxs;
    idxs.push_back(0);
    std::vector<std::pair<size_t, double> > expected_values;
    
    index = chains->get_total_param_index(chains->param_name_to_index("alpha0"),
					  idxs);
    expected_values.push_back(make_pair(index, 106.6));

    index = chains->get_total_param_index(chains->param_name_to_index("mu_beta"),
					  idxs);
    expected_values.push_back(make_pair(index, 6.185));
    
    index = chains->get_total_param_index(chains->param_name_to_index("sigma_y"),
					  idxs);
expected_values.push_back(make_pair(index, 6.086));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Rats,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Rats);
