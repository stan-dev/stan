#include <stan/model/log_prob_grad.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <test/test-models/good/model/parameters.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, write_array_unconstrain_array_roundtrip) {
  stan::io::empty_var_context data_var_context;
  stan_model model(data_var_context, 0, static_cast<std::stringstream*>(0));

  std::vector<std::string> json_path;
  json_path = {"src",  "test",  "test-models",
               "good", "model", "parameters.inits.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data inits(in);

  std::stringstream out;
  out.str("");

  // unused in this model but needed for write_array
  auto rng = stan::services::util::create_rng(12324232, 1);

  try {
    Eigen::VectorXd init_vector;
    model.transform_inits(inits, init_vector, &out);

    Eigen::VectorXd written_vector;
    model.write_array(rng, init_vector, written_vector, &out);

    Eigen::VectorXd recovered_vector;
    model.unconstrain_array(written_vector, recovered_vector, &out);

    EXPECT_MATRIX_NEAR(init_vector, recovered_vector, 1e-10);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "write_array_unconstrain_array_roundtrip Eigen::VectorXd";
  }

  try {
    std::vector<int> unused;
    std::vector<double> init_vector;
    model.transform_inits(inits, unused, init_vector, &out);

    std::vector<double> written_vector;
    model.write_array(rng, init_vector, unused, written_vector, &out);

    std::vector<double> recovered_vector;
    model.unconstrain_array(written_vector, recovered_vector, &out);

    for (int i = 0; i < init_vector.size(); i++) {
      EXPECT_NEAR(init_vector[i], recovered_vector[i], 1e-10);
    }

    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "write_array_unconstrain_array_roundtrip std::vector<double>";
  }
}
