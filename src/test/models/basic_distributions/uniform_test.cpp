#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_Uniform :
  public Model_Test_Fixture<Models_BasicDistributions_Uniform> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("uniform");
    return model_path;
  }
  static bool has_data() {
    return false;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations(int i) {
    std::vector<int> num_iter;
    num_iter.push_back(2000); //iterations for nuts
    num_iter.push_back(2000); //iterations for unit_metro
    num_iter.push_back(2000); //iterations for diag_metro
    num_iter.push_back(2000); //iterations for dense_metro
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

    expected_values.push_back(make_pair(chains[i]->index("y"), 0.5));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Uniform,
            Model_Test_Fixture,
            Models_BasicDistributions_Uniform);

TEST_F(Models_BasicDistributions_Uniform,
       Test_Variance) {
  populate_chains(0);
  
  int index = chains[0]->index("y");

  double sd = chains[0]->sd(index);
  double neff = chains[0]->effective_sample_size(index);
  
  EXPECT_NEAR(0.28867, sd, 1.96 * 1 / std::sqrt(2 * (neff - 1)));
}
