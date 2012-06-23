#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Salm2 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Salm2> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("salm");
    model_path.push_back("salm2");
    return model_path;
  }

  static bool has_data() {
    return true;
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    std::vector<std::pair<size_t, double> > expected_values;
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Salm2,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Salm2);
