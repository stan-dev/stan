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

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(4000); //iterations for nuts
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

    expected_values.push_back(make_pair(chains[i]->index("a[1]"), 0.712));
    expected_values.push_back(make_pair(chains[i]->index("a[2]"), 3.936));
    expected_values.push_back(make_pair(chains[i]->index("a[3]"), 5.28));

    expected_values.push_back(make_pair(chains[i]->index("beta"), 1.067));

    expected_values.push_back(make_pair(chains[i]->index("kappa"), 0.2463));

    expected_values.push_back(make_pair(chains[i]->index("log_sigma"), 0.195));

    expected_values.push_back(make_pair(chains[i]->index("pi"), -0.2367));

    expected_values.push_back(make_pair(chains[i]->index("sigma"), 1.24));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Inhalers,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Inhalers);
