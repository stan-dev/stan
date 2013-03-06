#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Cervix2 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Cervix2> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("cervix");
    model_path.push_back("cervix2");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
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

    expected_values.push_back(make_pair(chains->index("beta0C"), -0.921));

    expected_values.push_back(make_pair(chains->index("gamma1"), 0.4389));
    expected_values.push_back(make_pair(chains->index("gamma2"), 0.5964));

    expected_values.push_back(make_pair(chains->index("phi[1,1]"), 0.318));  // phi[1,1]
    expected_values.push_back(make_pair(chains->index("phi[1,2]"), 0.221));  // phi[1,2]
    expected_values.push_back(make_pair(chains->index("phi[2,1]"), 0.5664)); // phi[2,1]
    expected_values.push_back(make_pair(chains->index("phi[2,2]"), 0.7585)); // phi[2,2]

    expected_values.push_back(make_pair(chains->index("q"), 0.4953));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Cervix2,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Cervix2);
