#include <stan/run/model_config.hpp>
#include <gtest/gtest.h>

class ModelConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    json_dir_ = "src/test/unit/run/json/";
  }
  
  std::string json_dir_;
};

TEST_F(ModelConfigTest, DefaultConstructor) {
  auto config = stan::run::model_config::create()
    .data(json_dir_ + "bernoulli.data.json")
    .build();
  
  EXPECT_EQ(1u, config.seed());  // Default random seed
  EXPECT_NE(nullptr, config.data());
}

TEST_F(ModelConfigTest, ValidDataFile) {
  auto config = stan::run::model_config::create()
    .data(json_dir_ + "valid_data.json")
    .build();
  
  EXPECT_EQ(1u, config.seed());
  EXPECT_NE(nullptr, config.data());
}

TEST_F(ModelConfigTest, CustomSeed) {
  auto config = stan::run::model_config::create()
    .data(json_dir_ + "bernoulli.data.json")
    .seed(12345)
    .build();
  
  EXPECT_EQ(12345u, config.seed());
  EXPECT_NE(nullptr, config.data());
}

TEST_F(ModelConfigTest, NonexistentDataFile) {
  EXPECT_THROW(
    stan::run::model_config::create()
      .data(json_dir_ + "nonexistent_file.json")
      .build(),
    std::exception);
}

TEST_F(ModelConfigTest, EmptyDataFile) {
  EXPECT_THROW(
    stan::run::model_config::create()
      .data(json_dir_ + "empty_data.json")
      .build(),
    std::exception);
}

TEST_F(ModelConfigTest, InvalidDataFile) {
  EXPECT_THROW(
    stan::run::model_config::create()
      .data(json_dir_ + "invalid_data.json")
      .build(),
    std::exception);
}
