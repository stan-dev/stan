#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Pump : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Pump> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("pump");
    model_path.push_back("pump");
    return model_path;
  }

  static bool has_data() {
    return true;
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
    
    expected_values.push_back(make_pair( 0U, 0.6867));   // alpha
    expected_values.push_back(make_pair( 1U, 0.905024)); // beta
    expected_values.push_back(make_pair( 2U, 0.05986));  // theta[0]
    expected_values.push_back(make_pair( 3U, 0.1015));   // theta[1]
    expected_values.push_back(make_pair( 4U, 0.08899));  // theta[2]
    expected_values.push_back(make_pair( 5U, 0.1156));   // theta[3]
    expected_values.push_back(make_pair( 6U, 0.6043));   // theta[4]
    expected_values.push_back(make_pair( 7U, 0.6121));   // theta[5]
    expected_values.push_back(make_pair( 8U, 0.899));    // theta[6]
    expected_values.push_back(make_pair( 9U, 0.9095));   // theta[7]
    expected_values.push_back(make_pair(10U, 1.587));    // theta[8]
    expected_values.push_back(make_pair(11U, 1.995));    // theta[9]

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Pump,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Pump);
