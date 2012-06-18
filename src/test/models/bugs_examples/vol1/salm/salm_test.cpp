#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Salm : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Salm> {
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
    model_path.push_back("salm");
    return model_path;
  }

  static bool has_data() {
    return true;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Salm,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Salm);
