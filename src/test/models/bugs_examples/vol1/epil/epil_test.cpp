#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Epil : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Epil> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("epil");
    model_path.push_back("epil");
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
    num_iter.push_back(250000); //iterations for unit_metro
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
    
    expected_values.push_back(make_pair(chains[i]->index("alpha_Age"), 0.4891));
    expected_values.push_back(make_pair(chains[i]->index("alpha_BT"), 0.3496));
    expected_values.push_back(make_pair(chains[i]->index("alpha_Base"), 0.8931));
    expected_values.push_back(make_pair(chains[i]->index("alpha_Trt"), -0.9428));
    expected_values.push_back(make_pair(chains[i]->index("alpha_V4"), -0.1027));
    expected_values.push_back(make_pair(chains[i]->index("alpha0"), -1.435));
    expected_values.push_back(make_pair(chains[i]->index("sigma_b"), 0.3624));
    expected_values.push_back(make_pair(chains[i]->index("sigma_b1"), 0.4987));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Epil,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Epil);
