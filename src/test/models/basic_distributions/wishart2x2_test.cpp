#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

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

  static size_t num_iterations() {
    return iterations;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;

    expected_values.push_back(make_pair(5U, 2.0 * 4.0)); // W[1,1]
    expected_values.push_back(make_pair(6U, 0.0));       // W[1,2]
    expected_values.push_back(make_pair(7U, 0.0));       // W[2,1]
    expected_values.push_back(make_pair(8U, 0.5 * 4.0)); // W[2,2]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Wishart2x2,
			      Model_Test_Fixture,
			      Models_BasicDistributions_Wishart2x2);

TEST_F(Models_BasicDistributions_Wishart2x2,
       Test_Values) {
  populate_chains();
  std::vector<double> v1, v2;
  chains->get_samples(1U, v1);  // sd1
  chains->get_samples(5U, v2);  // W[1,1]
  for (size_t n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1[n]*v1[n], v2[n], 0.001)
      << "comparing sd1 to W[1,1]";
  }

  chains->get_samples(2U, v1);  // sd2
  chains->get_samples(8U, v2);  // W[2,2]
  for (size_t n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1[n]*v1[n], v2[n], 0.001)
      << "comparing sd2 to W[2,2]";
  }

  chains->get_samples(4U, v1);  // cov
  chains->get_samples(6U, v2);  // W[1,2]
  for (size_t n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1[n], v2[n], 0.001)
      << "comparing cov to W[1,2]";
  }

  chains->get_samples(6U, v1);  // W[1,2]
  chains->get_samples(7U, v2);  // W[2,1]
  for (size_t n = 0; n < chains->num_samples(); n++) {
    EXPECT_NEAR(v1[n], v2[n], 0.00001)
      << "comparing W[1,2] to W[2,1]";
  }

}

