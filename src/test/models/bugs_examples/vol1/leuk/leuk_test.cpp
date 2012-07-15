#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Leuk : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Leuk> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("leuk");
    model_path.push_back("leuk");
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

  static std::vector<size_t> skip_chains_test() {
    std::vector<size_t> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    size_t index;
    std::vector<size_t> dims;
    dims.push_back(0U);
    std::vector<std::pair<size_t, double> > expected_values;

    index = chains->get_total_param_index(chains->param_name_to_index("beta"),
					  dims);
    expected_values.push_back(make_pair(index, 1.538));

    index = chains->get_total_param_index(chains->param_name_to_index("S_placebo"),
					  dims);
    expected_values.push_back(make_pair(index +  0U, 0.9282));
    expected_values.push_back(make_pair(index +  1U, 0.8538));
    expected_values.push_back(make_pair(index +  2U, 0.8161));
    expected_values.push_back(make_pair(index +  3U, 0.7432));
    expected_values.push_back(make_pair(index +  4U, 0.6703));
    expected_values.push_back(make_pair(index +  5U, 0.5633));
    expected_values.push_back(make_pair(index +  6U, 0.5304));
    expected_values.push_back(make_pair(index +  7U, 0.4142));
    expected_values.push_back(make_pair(index +  8U, 0.3812));
    expected_values.push_back(make_pair(index +  9U, 0.32));
    expected_values.push_back(make_pair(index + 10U, 0.2583));
    expected_values.push_back(make_pair(index + 11U, 0.02257));
    expected_values.push_back(make_pair(index + 12U, 0.1956));
    expected_values.push_back(make_pair(index + 13U, 0.1656));
    expected_values.push_back(make_pair(index + 14U, 0.1398));
    expected_values.push_back(make_pair(index + 15U, 0.0867));
    expected_values.push_back(make_pair(index + 16U, 0.04445));

    index = chains->get_total_param_index(chains->param_name_to_index("S_treat"),
					  dims);
    expected_values.push_back(make_pair(index +  0U, 0.983));
    expected_values.push_back(make_pair(index +  1U, 0.9643));
    expected_values.push_back(make_pair(index +  2U, 0.9544));
    expected_values.push_back(make_pair(index +  3U, 0.9343));
    expected_values.push_back(make_pair(index +  4U, 0.9125));
    expected_values.push_back(make_pair(index +  5U, 0.8772));
    expected_values.push_back(make_pair(index +  6U, 0.8652));
    expected_values.push_back(make_pair(index +  7U, 0.8178));
    expected_values.push_back(make_pair(index +  8U, 0.8024));
    expected_values.push_back(make_pair(index +  9U, 0.771));
    expected_values.push_back(make_pair(index + 10U, 0.7339));
    expected_values.push_back(make_pair(index + 11U, 0.7114));
    expected_values.push_back(make_pair(index + 12U, 0.6882));
    expected_values.push_back(make_pair(index + 13U, 0.6619));
    expected_values.push_back(make_pair(index + 14U, 0.636));
    expected_values.push_back(make_pair(index + 15U, 0.5662));
    expected_values.push_back(make_pair(index + 16U, 0.4761));

    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BugsExamples_Vol1_Leuk,
			      Model_Test_Fixture,
			      Models_BugsExamples_Vol1_Leuk);
