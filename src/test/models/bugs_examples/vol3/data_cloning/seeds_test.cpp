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

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(2000); //iterations for nuts
    num_iter.push_back(200000); //iterations for unit_metro
    num_iter.push_back(200000); //iterations for diag_metro
    num_iter.push_back(200000); //iterations for dense_metro
    return num_iter[i];
  }

  static std::vector<int> skip_chains_test(int i) {
    std::vector<int> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains(int i) {
    default_populate_chains(i);
  }

  static std::vector<std::pair<int, double> >
  get_expected_values(int i) {
    using std::make_pair;
    std::vector<std::pair<int, double> > expected_values;

    // for k = 8
    expected_values.push_back(make_pair(chains[i]->index("alpha0"), -0.5516));  
    expected_values.push_back(make_pair(chains[i]->index("alpha1"), 0.1017)); 
    expected_values.push_back(make_pair(chains[i]->index("alpha12"), -0.8247));  
    expected_values.push_back(make_pair(chains[i]->index("alpha2"), 1.345)); 
    expected_values.push_back(make_pair(chains[i]->index("sigma"), 0.2407));  

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol3_Seeds,
            Model_Test_Fixture,
            Models_BugsExamples_Vol3_Seeds);
