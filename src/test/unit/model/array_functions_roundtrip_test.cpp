#include <stan/model/log_prob_grad.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <test/test-models/good/model/parameters.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

auto get_init_json() {
  std::vector<std::string> json_path = {
      "src", "test", "test-models", "good", "model", "parameters.inits.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  return std::unique_ptr<stan::io::var_context>(new stan::json::json_data(in));
}

class ModelArrayFunctionsRoundtripTest : public testing::Test {
 public:
  ModelArrayFunctionsRoundtripTest()
      : model(context, 0, nullptr), rng(12324232), inits(get_init_json()) {}

  stan::io::empty_var_context context;
  stan_model model;
  boost::ecuyer1988 rng;
  std::unique_ptr<stan::io::var_context> inits;
  std::stringstream out;

  /**
   * Test that the unconstrain_array function is the inverse of the
   * write_array function. This tests the Eigen overloads.
   *
   * This calls transform_inits, write_array, and then unconstrain_array
   * and asserts that the output of unconstrain_array is the same as the
   * output of transform_inits.
   */
  void eigen_round_trip(bool include_gq, bool include_tp) {
    Eigen::VectorXd init_vector;
    model.transform_inits(*inits, init_vector, &out);

    Eigen::VectorXd written_vector;
    model.write_array(rng, init_vector, written_vector, include_gq, include_tp,
                      &out);

    Eigen::VectorXd recovered_vector;
    model.unconstrain_array(written_vector, recovered_vector, &out);

    EXPECT_MATRIX_NEAR(init_vector, recovered_vector, 1e-10);
    EXPECT_EQ("", out.str());
  }

  /**
   * Same as eigen_round_trip but for the std::vector overloads
   */
  void std_vec_round_trip(bool include_gq, bool include_tp) {
    std::vector<int> unused;
    std::vector<double> init_vector;
    model.transform_inits(*inits, unused, init_vector, &out);

    std::vector<double> written_vector;
    model.write_array(rng, init_vector, unused, written_vector, include_gq,
                      include_tp, &out);

    std::vector<double> recovered_vector;
    model.unconstrain_array(written_vector, recovered_vector, &out);

    for (int i = 0; i < init_vector.size(); i++) {
      EXPECT_NEAR(init_vector[i], recovered_vector[i], 1e-10);
    }

    EXPECT_EQ("", out.str());
  }
};

// test all combinations of include_gq and include_tp.
// unconstrain_array should ignore them as they appear at the end
// of the written vectors

TEST_F(ModelArrayFunctionsRoundtripTest, eigen_overloads) {
  eigen_round_trip(false, false);
  eigen_round_trip(false, true);
  eigen_round_trip(true, false);
  eigen_round_trip(true, true);
}

TEST_F(ModelArrayFunctionsRoundtripTest, std_vector_overloads) {
  std_vec_round_trip(false, false);
  std_vec_round_trip(false, true);
  std_vec_round_trip(true, false);
  std_vec_round_trip(true, true);
}
