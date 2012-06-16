#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Simt4 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Simt4,
                                       true> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("t_df");
    model_path.push_back("simt4");
    return model_path;
  }

};

TEST_F(Models_BugsExamples_Vol2_Simt4,RunModel) {
  run_model();
}
