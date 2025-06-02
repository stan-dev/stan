#include <stan/run/load_model.hpp>
#include <stan/run/model_config.hpp>

#include <test/test-models/good/services/bernoulli.hpp>

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

TEST(LoadModelTest, LoadModelWithValidData) {
  auto config = stan::run::model_config::create()
    .data("src/test/test-models/good/services/bernoulli.data.json")
    .seed(12345)
    .build();

  auto& model = stan::run::load_model(config);

  EXPECT_FALSE(model.model_name().empty());
  EXPECT_EQ(model.model_name(), "bernoulli_model");
  EXPECT_EQ(model.num_params_r(), 1);
  
  std::vector<std::string> param_names;
  model.constrained_param_names(param_names, false, false);
  EXPECT_FALSE(param_names.empty());
  EXPECT_EQ(param_names[0], "theta");
  
  std::vector<std::string> uparam_names;
  model.unconstrained_param_names(uparam_names, false, false);
  EXPECT_FALSE(uparam_names.empty());
  EXPECT_EQ(uparam_names[0], "theta");
}

TEST(LoadModelTest, LoadModelWithNonexistentFile) {
  EXPECT_THROW({
    auto config = stan::run::model_config::create()
      .data("nonexistent_file.json")
      .seed(12345)
      .build();
  }, std::runtime_error);
}

TEST(LoadModelTest, LoadModelWithInvalidJSON) {
  EXPECT_THROW({
    auto config = stan::run::model_config::create()
      .data("src/test/json/invalid_data.json")
      .seed(12345)
      .build();
  }, std::runtime_error);
}
