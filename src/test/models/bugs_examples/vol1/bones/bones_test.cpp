#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Bones : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Bones> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("bones");
    model_path.push_back("bones");
    return model_path;
  }
  static bool has_data() {
    return true;
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
    expected_values.push_back(make_pair( 0U, 0.3244));
    expected_values.push_back(make_pair( 1U, 1.366));
    expected_values.push_back(make_pair( 2U, 2.357));
    expected_values.push_back(make_pair( 3U, 2.902));
    expected_values.push_back(make_pair( 4U, 5.535));
    expected_values.push_back(make_pair( 5U, 6.751));
    expected_values.push_back(make_pair( 6U, 6.451));
    expected_values.push_back(make_pair( 7U, 8.93));
    expected_values.push_back(make_pair( 8U, 8.981));
    expected_values.push_back(make_pair( 9U, 11.94));
    expected_values.push_back(make_pair(10U, 11.58));
    expected_values.push_back(make_pair(11U, 15.79));
    expected_values.push_back(make_pair(12U, 16.96));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Bones,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Bones);
