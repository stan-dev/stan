#include <stan/run/process_config.hpp>
#include <gtest/gtest.h>

class ProcessConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    json_dir_ = "src/test/unit/run/json/";
    output_dir_ = "test_output";
  }
  
  std::string json_dir_;
  std::string output_dir_;
};

TEST_F(ProcessConfigTest, DefaultConstructor) {
  auto config = stan::run::process_config::create()
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(1u, config.num_chains());
  EXPECT_EQ(1u, config.seed());
  EXPECT_DOUBLE_EQ(2.0, config.init_radius());
  EXPECT_EQ(output_dir_, config.output_dir());
  EXPECT_FALSE(config.has_init_params());
  EXPECT_TRUE(config.init_params().empty());
}

TEST_F(ProcessConfigTest, CustomValues) {
  auto config = stan::run::process_config::create()
    .num_chains(4)
    .seed(12345)
    .init_radius(1.5)
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(4u, config.num_chains());
  EXPECT_EQ(12345u, config.seed());
  EXPECT_DOUBLE_EQ(1.5, config.init_radius());
  EXPECT_EQ(output_dir_, config.output_dir());
  EXPECT_FALSE(config.has_init_params());
}

TEST_F(ProcessConfigTest, WithInitParams) {
  std::vector<std::string> init_files = {json_dir_ + "parameters.inits.json"};
  
  auto config = stan::run::process_config::create()
    .num_chains(1)
    .init_params(init_files)
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(1u, config.num_chains());
  EXPECT_TRUE(config.has_init_params());
  EXPECT_EQ(1u, config.init_params().size());
  EXPECT_NE(nullptr, config.init_params()[0]);
}

TEST_F(ProcessConfigTest, InvalidNumChains) {
  EXPECT_THROW(
    stan::run::process_config::create()
      .num_chains(0)
      .output_dir("test")
      .build(),
    std::invalid_argument);
}

TEST_F(ProcessConfigTest, InvalidInitRadius) {
  EXPECT_THROW(
    stan::run::process_config::create()
      .init_radius(0.0)
      .output_dir("test")
      .build(),
    std::invalid_argument);
  
  EXPECT_THROW(
    stan::run::process_config::create()
      .init_radius(-1.0)
      .output_dir("test")
      .build(),
    std::invalid_argument);
}

TEST_F(ProcessConfigTest, MismatchedInitFiles) {
  std::vector<std::string> init_files = {json_dir_ + "parameters.inits.json"};
  
  EXPECT_THROW(
    stan::run::process_config::create()
      .num_chains(3)  // 3 chains but only 1 init file
      .init_params(init_files)
      .output_dir(output_dir_)
      .build(),
    std::invalid_argument);
}

TEST_F(ProcessConfigTest, NonexistentInitFile) {
  std::vector<std::string> bad_files = {json_dir_ + "nonexistent.json"};
  
  EXPECT_THROW(
    stan::run::process_config::create()
      .num_chains(1)
      .init_params(bad_files)
      .output_dir(output_dir_)
      .build(),
    std::runtime_error);
}

TEST_F(ProcessConfigTest, InvalidInitFile) {
  std::vector<std::string> invalid_files = {json_dir_ + "invalid_data.json"};
  
  EXPECT_THROW(
    stan::run::process_config::create()
      .num_chains(1)
      .init_params(invalid_files)
      .output_dir(output_dir_)
      .build(),
    std::runtime_error);
}

TEST_F(ProcessConfigTest, EmptyInitFileList) {
  std::vector<std::string> empty_files = {};
  
  auto config = stan::run::process_config::create()
    .num_chains(2)
    .init_params(empty_files)
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(2u, config.num_chains());
  EXPECT_FALSE(config.has_init_params());
  EXPECT_TRUE(config.init_params().empty());
}

TEST_F(ProcessConfigTest, MultipleInitFiles) {
  // Create multiple init file references (same file used multiple times for test)
  std::vector<std::string> init_files = {
    json_dir_ + "parameters.inits.json",
    json_dir_ + "parameters.inits.json",
    json_dir_ + "parameters.inits.json"
  };
  
  auto config = stan::run::process_config::create()
    .num_chains(3)
    .init_params(init_files)
    .output_dir(output_dir_)
    .build();
  
  EXPECT_EQ(3u, config.num_chains());
  EXPECT_TRUE(config.has_init_params());
  EXPECT_EQ(3u, config.init_params().size());
  
  for (const auto& init_param : config.init_params()) {
    EXPECT_NE(nullptr, init_param);
  }
}
