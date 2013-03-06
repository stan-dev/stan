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

  static int num_iterations() {
    return iterations;
  }

  static std::vector<int> skip_chains_test() {
    std::vector<int> params_to_skip;
    params_to_skip.push_back(chains->index("alpha[1]"));

    params_to_skip.push_back(chains->index("beta[1,1]"));
    params_to_skip.push_back(chains->index("beta[1,2]"));
    params_to_skip.push_back(chains->index("beta[1,3]"));
    params_to_skip.push_back(chains->index("beta[1,4]"));
    params_to_skip.push_back(chains->index("beta[1,5]"));

    params_to_skip.push_back(chains->index("beta[1,1]"));
    params_to_skip.push_back(chains->index("beta[2,1]"));
    params_to_skip.push_back(chains->index("beta[3,1]"));
    params_to_skip.push_back(chains->index("beta[4,1]"));


    params_to_skip.push_back(chains->index("gamma[1,1]"));
    params_to_skip.push_back(chains->index("gamma[1,2]"));
    params_to_skip.push_back(chains->index("gamma[1,3]"));
    params_to_skip.push_back(chains->index("gamma[1,4]"));
    params_to_skip.push_back(chains->index("gamma[1,5]"));

    params_to_skip.push_back(chains->index("gamma[1,1]"));
    params_to_skip.push_back(chains->index("gamma[2,1]"));

    
    params_to_skip.push_back(chains->index("b[1,1]"));
    params_to_skip.push_back(chains->index("b[1,2]"));
    params_to_skip.push_back(chains->index("b[1,3]"));
    params_to_skip.push_back(chains->index("b[1,4]"));
    params_to_skip.push_back(chains->index("b[1,5]"));
    params_to_skip.push_back(chains->index("b[2,1]"));
    params_to_skip.push_back(chains->index("b[2,2]"));
    params_to_skip.push_back(chains->index("b[2,3]"));
    params_to_skip.push_back(chains->index("b[2,4]"));
    params_to_skip.push_back(chains->index("b[2,5]"));
    params_to_skip.push_back(chains->index("b[3,1]"));
    params_to_skip.push_back(chains->index("b[3,2]"));
    params_to_skip.push_back(chains->index("b[3,3]"));
    params_to_skip.push_back(chains->index("b[3,4]"));
    params_to_skip.push_back(chains->index("b[3,5]"));
    params_to_skip.push_back(chains->index("b[4,1]"));
    params_to_skip.push_back(chains->index("b[4,2]"));
    params_to_skip.push_back(chains->index("b[4,3]"));
    params_to_skip.push_back(chains->index("b[4,4]"));
    params_to_skip.push_back(chains->index("b[4,5]"));

    params_to_skip.push_back(chains->index("g[1,1]"));
    params_to_skip.push_back(chains->index("g[1,2]"));
    params_to_skip.push_back(chains->index("g[1,3]"));
    params_to_skip.push_back(chains->index("g[1,4]"));
    params_to_skip.push_back(chains->index("g[1,5]"));
    params_to_skip.push_back(chains->index("g[2,1]"));
    params_to_skip.push_back(chains->index("g[2,2]"));
    params_to_skip.push_back(chains->index("g[2,3]"));
    params_to_skip.push_back(chains->index("g[2,4]"));
    params_to_skip.push_back(chains->index("g[2,5]"));

    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("b[1,2]"), -1.876));  // b[1,2]
    expected_values.push_back(make_pair(chains->index("b[1,3]"), -0.3569)); // b[1,3]
    expected_values.push_back(make_pair(chains->index("b[1,4]"), 0.5489));  // b[1,4]
    expected_values.push_back(make_pair(chains->index("b[1,5]"), 0.272));   // b[1,5]

    expected_values.push_back(make_pair(chains->index("b[2,2]"), 0.8931));  // b[2,2]
    expected_values.push_back(make_pair(chains->index("b[2,3]"), 0.9655));  // b[2,3]
    expected_values.push_back(make_pair(chains->index("b[2,4]"), -1.252));  // b[2,4]
    expected_values.push_back(make_pair(chains->index("b[2,5]"), -0.6444)); // b[2,5]

    expected_values.push_back(make_pair(chains->index("b[3,2]"), 1.075));   // b[3,2]
    expected_values.push_back(make_pair(chains->index("b[3,3]"), 1.445));   // b[3,3]
    expected_values.push_back(make_pair(chains->index("b[3,4]"), 0.9271));  // b[3,4]
    expected_values.push_back(make_pair(chains->index("b[3,5]"), 0.9803));  // b[3,5]

    expected_values.push_back(make_pair(chains->index("b[4,2]"), -0.0924));  // b[4,2]
    expected_values.push_back(make_pair(chains->index("b[4,3]"), -2.054));   // b[4,3]
    expected_values.push_back(make_pair(chains->index("b[4,4]"), -0.2242));  // b[4,4]
    expected_values.push_back(make_pair(chains->index("b[4,5]"), -0.6079));  // b[4,5]


    expected_values.push_back(make_pair(chains->index("g[1,2]"), 0.7647)); // g[1,2]
    expected_values.push_back(make_pair(chains->index("g[1,3]"), -0.176)); // g[1,3]
    expected_values.push_back(make_pair(chains->index("g[1,4]"), -0.3441));// g[1,4]
    expected_values.push_back(make_pair(chains->index("g[1,5]"), 0.1835)); // g[1,5]

    expected_values.push_back(make_pair(chains->index("g[2,2]"), -0.7647));// g[2,2]
    expected_values.push_back(make_pair(chains->index("g[2,3]"), 0.176));  // g[2,3]
    expected_values.push_back(make_pair(chains->index("g[2,4]"), 0.3441)); // g[2,4]
    expected_values.push_back(make_pair(chains->index("g[2,5]"), -0.1835));// g[2,5]
    
    return expected_values; 
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Alli2,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Alli2);
