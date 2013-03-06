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

  static int num_iterations() {
    return 8000;
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
    double nu = 10;
    
    expected_values.push_back(make_pair(chains->index("W[1,1]"), 2.9983662 * nu)); // W[1,1]
    expected_values.push_back(make_pair(chains->index("W[1,2]"), 0.2898776 * nu)); // W[1,2]
    expected_values.push_back(make_pair(chains->index("W[1,3]"), -2.650523 * nu)); // W[1,3]
    expected_values.push_back(make_pair(chains->index("W[1,4]"), 0.1055911 * nu)); // W[1,4]
    expected_values.push_back(make_pair(chains->index("W[2,1]"), 0.2898776 * nu)); // W[2,1]
    expected_values.push_back(make_pair(chains->index("W[2,2]"), 11.4803610 * nu));// W[2,2]
    expected_values.push_back(make_pair(chains->index("W[2,3]"), 7.157993 * nu));  // W[2,3]
    expected_values.push_back(make_pair(chains->index("W[2,4]"), -3.1129955 * nu));// W[2,4]
    expected_values.push_back(make_pair(chains->index("W[3,1]"), -2.650523 * nu)); // W[3,1]
    expected_values.push_back(make_pair(chains->index("W[3,2]"), 7.157993 * nu));  // W[3,2]
    expected_values.push_back(make_pair(chains->index("W[3,3]"), 11.676181 * nu)); // W[3,3]
    expected_values.push_back(make_pair(chains->index("W[3,4]"), -3.5866852 * nu));// W[3,4]
    expected_values.push_back(make_pair(chains->index("W[4,1]"), 0.1055911 * nu)); // W[4,1]
    expected_values.push_back(make_pair(chains->index("W[4,2]"), -3.1129955 * nu));// W[4,2]
    expected_values.push_back(make_pair(chains->index("W[4,3]"), -3.5866852 * nu));// W[4,3]
    expected_values.push_back(make_pair(chains->index("W[4,4]"), 1.4482736 * nu)); // W[4,4]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Wishart2,
            Model_Test_Fixture,
            Models_BasicDistributions_Wishart2);
