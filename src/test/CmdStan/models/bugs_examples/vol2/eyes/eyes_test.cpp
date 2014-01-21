#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Eyes : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Eyes> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("eyes");
    model_path.push_back("eyes");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return true;
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
    std::vector<int> dims;
    dims.push_back(0);

    std::vector<std::pair<int, double> > expected_values;

    expected_values.push_back(make_pair(chains->index("p1"), 0.6014));

    expected_values.push_back(make_pair(chains->index("lambda[1]"), 536.8));
    expected_values.push_back(make_pair(chains->index("lambda[2]"), 548.9));

    expected_values.push_back(make_pair(chains->index("sigma"), 3.805));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol2_Eyes,
            Model_Test_Fixture,
            Models_BugsExamples_Vol2_Eyes);
