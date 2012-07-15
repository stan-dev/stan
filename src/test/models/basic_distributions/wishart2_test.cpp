#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_Wishart2 :
  public Model_Test_Fixture<Models_BasicDistributions_Wishart2> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("wishart2");
    return model_path;
  }
  static bool has_data() {
    return false;
  }

  static bool has_init() {
    return false;
  }

  static size_t num_iterations() {
    return 8000U;
  }

  static std::vector<size_t> skip_chains_test() {
    std::vector<size_t> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;
    double nu = 10;
    
    expected_values.push_back(make_pair( 0U, 2.9983662 * nu)); // W[1,1]
    expected_values.push_back(make_pair( 1U, 0.2898776 * nu)); // W[1,2]
    expected_values.push_back(make_pair( 2U, -2.650523 * nu)); // W[1,3]
    expected_values.push_back(make_pair( 3U, 0.1055911 * nu)); // W[1,4]
    expected_values.push_back(make_pair( 4U, 0.2898776 * nu)); // W[2,1]
    expected_values.push_back(make_pair( 5U, 11.4803610 * nu));// W[2,2]
    expected_values.push_back(make_pair( 6U, 7.157993 * nu));  // W[2,3]
    expected_values.push_back(make_pair( 7U, -3.1129955 * nu));// W[2,4]
    expected_values.push_back(make_pair( 8U, -2.650523 * nu)); // W[3,1]
    expected_values.push_back(make_pair( 9U, 7.157993 * nu));  // W[3,2]
    expected_values.push_back(make_pair(10U, 11.676181 * nu)); // W[3,3]
    expected_values.push_back(make_pair(11U, -3.5866852 * nu));// W[3,4]
    expected_values.push_back(make_pair(12U, 0.1055911 * nu)); // W[4,1]
    expected_values.push_back(make_pair(13U, -3.1129955 * nu));// W[4,2]
    expected_values.push_back(make_pair(14U, -3.5866852 * nu));// W[4,3]
    expected_values.push_back(make_pair(15U, 1.4482736 * nu)); // W[4,4]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Wishart2,
			      Model_Test_Fixture,
			      Models_BasicDistributions_Wishart2);
