#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Lsat : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Lsat> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("lsat");
    model_path.push_back("lsat");
    return model_path;
  }

  static bool has_data() {
    return true;
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

    expected_values.push_back(make_pair(chains->index("a[1]"), -1.26));
    expected_values.push_back(make_pair(chains->index("a[2]"), 0.4776));
    expected_values.push_back(make_pair(chains->index("a[3]"), 1.239));
    expected_values.push_back(make_pair(chains->index("a[4]"), 0.1696));
    expected_values.push_back(make_pair(chains->index("a[5]"), -0.6256));

    expected_values.push_back(make_pair(chains->index("beta"), 0.7582));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Lsat,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Lsat);
