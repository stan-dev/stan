#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Surgical : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Surgical> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("surgical");
    model_path.push_back("surgical");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
  }

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(2000); //iterations for nuts
    num_iter.push_back(5000); //iterations for unit_metro
    num_iter.push_back(5000); //iterations for diag_metro
    num_iter.push_back(5000); //iterations for dense_metro
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

    expected_values.push_back(make_pair(chains[i]->index("mu"), -2.558));   // mu
    expected_values.push_back(make_pair(chains[i]->index("p[1]"), 0.05302));  // p[1]
    expected_values.push_back(make_pair(chains[i]->index("p[2]"), 0.1029));   // p[2]
    expected_values.push_back(make_pair(chains[i]->index("p[3]"), 0.07044));  // p[3]
    expected_values.push_back(make_pair(chains[i]->index("p[4]"), 0.0593));   // p[4]
    expected_values.push_back(make_pair(chains[i]->index("p[5]"), 0.05187));  // p[5]
    expected_values.push_back(make_pair(chains[i]->index("p[6]"), 0.06903));  // p[6]
    expected_values.push_back(make_pair(chains[i]->index("p[7]"), 0.06682));  // p[7]
    expected_values.push_back(make_pair(chains[i]->index("p[8]"), 0.1226));   // p[8]
    expected_values.push_back(make_pair(chains[i]->index("p[9]"), 0.0698));   // p[9]
    expected_values.push_back(make_pair(chains[i]->index("p[10]"), 0.07851));  // p[10]
    expected_values.push_back(make_pair(chains[i]->index("p[11]"), 0.1021));   // p[11]
    expected_values.push_back(make_pair(chains[i]->index("p[2]"), 0.06858));  // p[12]
    expected_values.push_back(make_pair(chains[i]->index("pop_mean"), 0.07259));  // pop.mean
    expected_values.push_back(make_pair(chains[i]->index("sigma"), 0.4028));   // sigma

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Surgical,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Surgical);
