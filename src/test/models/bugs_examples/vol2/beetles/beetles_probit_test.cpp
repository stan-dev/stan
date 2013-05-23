#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_BeetlesProbit : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_BeetlesProbit> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("beetles");
    model_path.push_back("beetles_probit");
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

    expected_values.push_back(make_pair(chains[i]->index("alpha"), -35.04));

    expected_values.push_back(make_pair(chains[i]->index("beta"), 19.79));

    expected_values.push_back(make_pair(chains[i]->index("rhat[1]"), 3.442));
    expected_values.push_back(make_pair(chains[i]->index("rhat[2]"), 10.76));
    expected_values.push_back(make_pair(chains[i]->index("rhat[3]"), 23.48));
    expected_values.push_back(make_pair(chains[i]->index("rhat[4]"), 33.81));
    expected_values.push_back(make_pair(chains[i]->index("rhat[5]"), 49.6));
    expected_values.push_back(make_pair(chains[i]->index("rhat[6]"), 53.27));
    expected_values.push_back(make_pair(chains[i]->index("rhat[7]"), 59.6));
    expected_values.push_back(make_pair(chains[i]->index("rhat[8]"), 59.17));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_BeetlesProbit,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_BeetlesProbit);
