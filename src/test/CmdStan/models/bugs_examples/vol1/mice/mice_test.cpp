#include <gtest/gtest.h>
#include <test/CmdStan/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Mice : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Mice> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("mice");
    model_path.push_back("mice");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static bool has_init() {
    return false;
  }

  static int num_iterations() {
    return 4000;
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

    expected_values.push_back(make_pair(chains->index("median[1]"), 23.65));
    expected_values.push_back(make_pair(chains->index("median[2]"), 35.18));
    expected_values.push_back(make_pair(chains->index("median[3]"), 26.68));
    expected_values.push_back(make_pair(chains->index("median[4]"), 21.28));

    expected_values.push_back(make_pair(chains->index("pos_control"), 0.3088));

    expected_values.push_back(make_pair(chains->index("r"), 2.902));

    expected_values.push_back(make_pair(chains->index("test_sub"), -0.3475));
    
    expected_values.push_back(make_pair(chains->index("veh_control"), -1.143));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Mice,
            Model_Test_Fixture,
            Models_BugsExamples_Vol1_Mice);
