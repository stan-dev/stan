#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>
#include <boost/math/distributions/students_t.hpp>

class Models_BasicDistributions_Binormal : 
  public Model_Test_Fixture<Models_BasicDistributions_Binormal> {
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("binormal");
    return model_path;
  }
    
  static bool has_data() {
    return false;
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;
    size_t index;
    std::vector<size_t> idxs;
    
    idxs.push_back(0);
    index = chains->get_total_param_index
      (chains->param_name_to_index("y"),idxs);

    // y1 should be 0.0
    expected_values.push_back(make_pair(index, 0.0));
    
    idxs.clear();
    idxs.push_back(1);
    index = chains->get_total_param_index
      (chains->param_name_to_index("y"),idxs);
    
    // y2 should be 0.0
    expected_values.push_back(make_pair(index, 0.0));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Binormal,
			      Model_Test_Fixture,
			      Models_BasicDistributions_Binormal);

