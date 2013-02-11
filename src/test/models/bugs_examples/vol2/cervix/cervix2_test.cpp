#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Cervix2 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Cervix2> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("cervix");
    model_path.push_back("cervix2");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
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
    dims.push_back(0);

    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("beta0C"),
					  dims);
    expected_values.push_back(make_pair(index, -0.921));

    index = chains->get_total_param_index(chains->param_name_to_index("gamma1"),
					  dims);
    expected_values.push_back(make_pair(index, 0.4389));
    index = chains->get_total_param_index(chains->param_name_to_index("gamma2"),
					  dims);
    expected_values.push_back(make_pair(index, 0.5964));

    dims.push_back(0);
    index = chains->get_total_param_index(chains->param_name_to_index("phi"),
					  dims);
    expected_values.push_back(make_pair(index + 0U, 0.318));  // phi[1,1]
    expected_values.push_back(make_pair(index + 2U, 0.221));  // phi[1,2]
    expected_values.push_back(make_pair(index + 1U, 0.5664)); // phi[2,1]
    expected_values.push_back(make_pair(index + 3U, 0.7585)); // phi[2,2]

    dims.resize(1);
    index = chains->get_total_param_index(chains->param_name_to_index("q"),
					  dims);
    expected_values.push_back(make_pair(index, 0.4953));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Cervix2,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol2_Cervix2);
