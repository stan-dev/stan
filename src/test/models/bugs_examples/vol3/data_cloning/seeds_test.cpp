#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol3_Seeds : 
  public Model_Test_Fixture<Models_BugsExamples_Vol3_Seeds> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol3");
    model_path.push_back("data_cloning");
    model_path.push_back("seeds");
    return model_path;
  }
  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations() {
    return iterations;
  }

  static std::vector<int> skip_chains_test() {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<int, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    // for k = 8
    expected_values.push_back(make_pair(chains->index("alpha0"), -0.5516));  
    expected_values.push_back(make_pair(chains->index("alpha1"), 0.1017)); 
    expected_values.push_back(make_pair(chains->index("alpha12"), -0.8247));  
    expected_values.push_back(make_pair(chains->index("alpha2"), 1.345)); 
    expected_values.push_back(make_pair(chains->index("sigma"), 0.2407));  

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol3_Seeds,
            Model_Test_Fixture,
            Models_BugsExamples_Vol3_Seeds);
