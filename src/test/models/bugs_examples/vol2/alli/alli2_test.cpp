#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Alli2 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Alli2> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("alli");
    model_path.push_back("alli2");
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
    size_t I = 4, J = 2, K = 5;
    std::vector<size_t> params_to_skip;
    size_t index;
    std::vector<size_t> dims;
    dims.push_back(0);
    index = chains->get_total_param_index(chains->param_name_to_index("alpha"),
					  dims);
    params_to_skip.push_back(index);

    dims.push_back(0);
    dims[0] = 0;
    for (size_t k = 0; k < K; k++)  {
      dims[1] = k;
      index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					    dims);
      params_to_skip.push_back(index);
    }
    dims[1] = 0;
    for (size_t i = 1; i < I; i++) {
      dims[0] = i;
      index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					    dims);
      params_to_skip.push_back(index);
      }


    dims[0] = 0;
    for (size_t k = 0; k < K; k++)  {
      dims[1] = k;
      index = chains->get_total_param_index(chains->param_name_to_index("gamma"),
					    dims);
      params_to_skip.push_back(index);
    }
    dims[1] = 0;
    for (size_t j = 1; j < J; j++) {
      dims[0] = j;
      index = chains->get_total_param_index(chains->param_name_to_index("gamma"),
					    dims);
      params_to_skip.push_back(index);
    }

    dims[0] = 0;
    dims[1] = 0;
    index = chains->get_total_param_index(chains->param_name_to_index("b"),
					  dims);
    for (size_t i = 0; i < I; i++) 
      for (size_t k = 0; k < K; k++)
	params_to_skip.push_back(index++);

    dims[0] = 0;
    dims[1] = 0;
    index = chains->get_total_param_index(chains->param_name_to_index("g"),
					  dims);
    for (size_t j = 0; j < J; j++) 
      for (size_t k = 0; k < K; k++)
	params_to_skip.push_back(index++);


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

    index = chains->get_total_param_index(chains->param_name_to_index("b"),
					  dims);

    expected_values.push_back(make_pair(index + 4U, -1.876));  // b[1,2]
    expected_values.push_back(make_pair(index + 8U, -0.3569)); // b[1,3]
    expected_values.push_back(make_pair(index +12U, 0.5489));  // b[1,4]
    expected_values.push_back(make_pair(index +16U, 0.272));   // b[1,5]

    expected_values.push_back(make_pair(index + 5U, 0.8931));  // b[2,2]
    expected_values.push_back(make_pair(index + 9U, 0.9655));  // b[2,3]
    expected_values.push_back(make_pair(index +13U, -1.252));  // b[2,4]
    expected_values.push_back(make_pair(index +17U, -0.6444)); // b[2,5]

    expected_values.push_back(make_pair(index + 6U, 1.075));   // b[3,2]
    expected_values.push_back(make_pair(index +10U, 1.445));   // b[3,3]
    expected_values.push_back(make_pair(index +14U, 0.9271));  // b[3,4]
    expected_values.push_back(make_pair(index +18U, 0.9803));  // b[3,5]


    expected_values.push_back(make_pair(index + 7U, -0.0924));  // b[4,2]
    expected_values.push_back(make_pair(index +11U, -2.054));   // b[4,3]
    expected_values.push_back(make_pair(index +15U, -0.2242));  // b[4,4]
    expected_values.push_back(make_pair(index +19U, -0.6079));  // b[4,5]


    index = chains->get_total_param_index(chains->param_name_to_index("g"),
					  dims);
    expected_values.push_back(make_pair(index + 2U, 0.7647)); // g[1,2]
    expected_values.push_back(make_pair(index + 4U, -0.176)); // g[1,3]
    expected_values.push_back(make_pair(index + 6U, -0.3441));// g[1,4]
    expected_values.push_back(make_pair(index + 8U, 0.1835)); // g[1,5]

    expected_values.push_back(make_pair(index + 3U, -0.7647));// g[2,2]
    expected_values.push_back(make_pair(index + 5U, 0.176));  // g[2,3]
    expected_values.push_back(make_pair(index + 7U, 0.3441)); // g[2,4]
    expected_values.push_back(make_pair(index + 9U, -0.1835));// g[2,5]
    
    
    return expected_values; 
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Alli2,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol2_Alli2);
