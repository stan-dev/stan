#include <gtest/gtest.h>
#include <test/stat-validity/stat_valid_test_fixture.hpp>

class Funnel :
  public Stat_Valid_Test_Fixture<Funnel> {
  
protected:
  virtual void SetUp() {}
    
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("src");
    model_path.push_back("test");
    model_path.push_back("test-models");
    model_path.push_back("compiled");
    model_path.push_back("stat-validity");
    model_path.push_back("funnel");
    return model_path;
  }

  static bool has_data() { return false; }
  static bool has_init() { return false; }
  static int num_warmup() { return 500; }
  static int num_samples() { return 50000; }

  static std::vector<std::pair<std::string, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<std::string, double> > expected_values;
    expected_values.push_back(make_pair("v", 0));
    expected_values.push_back(make_pair("v2", 9));
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Funnel,
                              Stat_Valid_Test_Fixture,
                              Funnel);
