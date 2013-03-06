#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Leuk : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Leuk> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("leuk");
    model_path.push_back("leuk");
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

    expected_values.push_back(make_pair(chains->index("beta"), 1.538));

    expected_values.push_back(make_pair(chains->index("S_placebo[1]"), 0.9282));
    expected_values.push_back(make_pair(chains->index("S_placebo[2]"), 0.8538));
    expected_values.push_back(make_pair(chains->index("S_placebo[3]"), 0.8161));
    expected_values.push_back(make_pair(chains->index("S_placebo[4]"), 0.7432));
    expected_values.push_back(make_pair(chains->index("S_placebo[5]"), 0.6703));
    expected_values.push_back(make_pair(chains->index("S_placebo[6]"), 0.5633));
    expected_values.push_back(make_pair(chains->index("S_placebo[7]"), 0.5304));
    expected_values.push_back(make_pair(chains->index("S_placebo[8]"), 0.4142));
    expected_values.push_back(make_pair(chains->index("S_placebo[9]"), 0.3812));
    expected_values.push_back(make_pair(chains->index("S_placebo[10]"), 0.32));
    expected_values.push_back(make_pair(chains->index("S_placebo[11]"), 0.2583));
    expected_values.push_back(make_pair(chains->index("S_placebo[12]"), 0.02257));
    expected_values.push_back(make_pair(chains->index("S_placebo[13]"), 0.1956));
    expected_values.push_back(make_pair(chains->index("S_placebo[14]"), 0.1656));
    expected_values.push_back(make_pair(chains->index("S_placebo[15]"), 0.1398));
    expected_values.push_back(make_pair(chains->index("S_placebo[16]"), 0.0867));
    expected_values.push_back(make_pair(chains->index("S_placebo[17]"), 0.04445));

    expected_values.push_back(make_pair(chains->index("S_treat[1]"), 0.983));
    expected_values.push_back(make_pair(chains->index("S_treat[2]"), 0.9643));
    expected_values.push_back(make_pair(chains->index("S_treat[3]"), 0.9544));
    expected_values.push_back(make_pair(chains->index("S_treat[4]"), 0.9343));
    expected_values.push_back(make_pair(chains->index("S_treat[5]"), 0.9125));
    expected_values.push_back(make_pair(chains->index("S_treat[6]"), 0.8772));
    expected_values.push_back(make_pair(chains->index("S_treat[7]"), 0.8652));
    expected_values.push_back(make_pair(chains->index("S_treat[8]"), 0.8178));
    expected_values.push_back(make_pair(chains->index("S_treat[9]"), 0.8024));
    expected_values.push_back(make_pair(chains->index("S_treat[10]"), 0.771));
    expected_values.push_back(make_pair(chains->index("S_treat[11]"), 0.7339));
    expected_values.push_back(make_pair(chains->index("S_treat[12]"), 0.7114));
    expected_values.push_back(make_pair(chains->index("S_treat[13]"), 0.6882));
    expected_values.push_back(make_pair(chains->index("S_treat[14]"), 0.6619));
    expected_values.push_back(make_pair(chains->index("S_treat[15]"), 0.636));
    expected_values.push_back(make_pair(chains->index("S_treat[16]"), 0.5662));
    expected_values.push_back(make_pair(chains->index("S_treat[17]"), 0.4761));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Leuk,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Leuk);
