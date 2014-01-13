#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Dugongs : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Dugongs> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("dugongs");
    model_path.push_back("dugongs");
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

    expected_values.push_back(make_pair(chains->index("U3"), 1.861));  // U3
    expected_values.push_back(make_pair(chains->index("alpha"), 2.652));  // alpha
    expected_values.push_back(make_pair(chains->index("beta"), 0.9729)); // beta
    expected_values.push_back(make_pair(chains->index("lambda"), 0.8623)); // lambda
    expected_values.push_back(make_pair(chains->index("sigma"), 0.0992)); // sigma

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Dugongs,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Dugongs);
