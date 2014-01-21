#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BasicDistributions_Wishart2x2 :
  public Model_Test_Fixture<Models_BasicDistributions_Wishart2x2> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("wishart2x2");
    return model_path;
  }
  static bool has_data() {
    return false;
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

    expected_values.push_back(make_pair(chains->index("W[1,1]"), 2.0 * 4.0)); // W[1,1]
    expected_values.push_back(make_pair(chains->index("W[1,2]"), 0.0));       // W[1,2]
    expected_values.push_back(make_pair(chains->index("W[2,1]"), 0.0));       // W[2,1]
    expected_values.push_back(make_pair(chains->index("W[2,2]"), 0.5 * 4.0)); // W[2,2]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Wishart2x2,
            Model_Test_Fixture,
            Models_BasicDistributions_Wishart2x2);

TEST_F(Models_BasicDistributions_Wishart2x2,
       Test_Values) {
  populate_chains();
  Eigen::VectorXd v1, v2;
  v1 = chains->samples("sd1");  // sd1
  v2 = chains->samples("W[1,1]");  // W[1,1]
  for (int n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1(n)*v1(n), v2(n), 0.001)
      << "comparing sd1 to W[1,1]";
  }

  v1 = chains->samples("sd2");  // sd2
  v2 = chains->samples("W[2,2]");  // W[2,2]
  for (int n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1(n)*v1(n), v2(n), 0.001)
      << "comparing sd2 to W[2,2]";
  }

  v1 = chains->samples("cov");  // cov
  v2 = chains->samples("W[1,2]");  // W[1,2]
  for (int n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1(n), v2(n), 0.001)
      << "comparing cov to W[1,2]";
  }

  v1 = chains->samples("W[1,2]");  // W[1,2]
  v2 = chains->samples("W[2,1]");  // W[2,1]
  for (int n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1(n), v2(n), 0.00001)
      << "comparing W[1,2] to W[2,1]";
  }

}

