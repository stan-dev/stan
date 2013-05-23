#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Stacks_D_Normal_Ridge : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Stacks_D_Normal_Ridge> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("stacks");
    model_path.push_back("stacks_d_normal_ridge");
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
    int index;
    std::vector<int> dims;
    dims.push_back(0);

    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains[i]->index("b[1]"), 0.6816));
    expected_values.push_back(make_pair(chains[i]->index("b[2]"), 1.1317));
    expected_values.push_back(make_pair(chains[i]->index("b[3]"), -0.1266));

    expected_values.push_back(make_pair(chains[i]->index("b0"), -40.53));

    expected_values.push_back(make_pair(chains[i]->index("sigma"), 3.395));

    expected_values.push_back(make_pair(chains[i]->index("outlier_3"), 0.0189));
    expected_values.push_back(make_pair(chains[i]->index("outlier_4"), 0.0471));
    expected_values.push_back(make_pair(chains[i]->index("outlier_21"), 0.2795));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Stacks_D_Normal_Ridge,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Stacks_D_Normal_Ridge);
