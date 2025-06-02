#include <stan/run/load_samplers.hpp>
#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/read_json_data.hpp>

#include <test/test-models/good/services/bernoulli.hpp>

#include <stan/callbacks/stream_logger.hpp>

#include <memory>
#include <vector>

#include <gtest/gtest.h>

class LoadSamplersTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Load test data and create model
    auto data_context = stan::run::read_json_data("src/test/test-models/good/services/bernoulli.data.json");
    model_ = std::make_unique<bernoulli_model_namespace::bernoulli_model>(*data_context, 12345);
    
    // Setup basic configuration using unique_ptr
    config_ = std::make_unique<stan::run::hmc_nuts_config>(
      stan::run::hmc_nuts_config::create()
        .num_chains(2)
        .seed(12345)
        .metric_type(stan::run::metric_t::DIAG_E)
        .stepsize(1.0)
        .max_depth(10)
        .delta(0.8)
        .build()
    );
    
    // Create empty init contexts (use default initialization)
    for (size_t i = 0; i < config_->num_chains(); ++i) {
      init_contexts_.push_back(stan::run::read_json_data(""));
      metric_contexts_.push_back(stan::run::read_json_data(""));
      init_writers_.push_back(nullptr);
    }
    
    logger_ = std::make_unique<stan::callbacks::stream_logger>(
      std::cout, std::cout, std::cout, std::cerr, std::cerr);
  }
  
  std::unique_ptr<bernoulli_model_namespace::bernoulli_model> model_;
  std::unique_ptr<stan::run::hmc_nuts_config> config_;  // Use unique_ptr to avoid default constructor requirement
  std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts_;
  std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts_;
  std::vector<stan::callbacks::writer*> init_writers_;
  std::unique_ptr<stan::callbacks::stream_logger> logger_;
};

TEST_F(LoadSamplersTest, CreateSamplers_DiagE) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .metric_type(stan::run::metric_t::DIAG_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
  
  auto sampler_configs = stan::run::create_samplers(*model_, config, init_contexts_, 
                                                   metric_contexts_, *logger_, init_writers_);
  
  // Should return a variant - we can't easily check the exact type, 
  // but we can verify it doesn't throw
  EXPECT_NO_THROW({
    std::visit([&config](auto& sampler_config) {
      EXPECT_EQ(sampler_config.samplers.size(), config.num_chains());
      EXPECT_EQ(sampler_config.rngs.size(), config.num_chains());
      EXPECT_EQ(sampler_config.init_params.size(), config.num_chains());
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, CreateSamplers_UnitE) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .metric_type(stan::run::metric_t::UNIT_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
  
  EXPECT_NO_THROW({
    auto sampler_configs = stan::run::create_samplers(*model_, config, init_contexts_, 
                                                     metric_contexts_, *logger_, init_writers_);
    
    std::visit([&config](auto& sampler_config) {
      EXPECT_EQ(sampler_config.samplers.size(), config.num_chains());
      EXPECT_EQ(sampler_config.rngs.size(), config.num_chains());
      EXPECT_EQ(sampler_config.init_params.size(), config.num_chains());
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, CreateSamplers_DenseE) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .metric_type(stan::run::metric_t::DENSE_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
  
  EXPECT_NO_THROW({
    auto sampler_configs = stan::run::create_samplers(*model_, config, init_contexts_, 
                                                     metric_contexts_, *logger_, init_writers_);
    
    std::visit([&config](auto& sampler_config) {
      EXPECT_EQ(sampler_config.samplers.size(), config.num_chains());
      EXPECT_EQ(sampler_config.rngs.size(), config.num_chains());
      EXPECT_EQ(sampler_config.init_params.size(), config.num_chains());
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, LoadSamplers_DiagE_CorrectSizes) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .metric_type(stan::run::metric_t::DIAG_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
    
  auto sampler_config = stan::run::load_samplers<stan::run::metric_t::DIAG_E>(
    *model_, config, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(sampler_config.samplers.size(), config.num_chains());
  EXPECT_EQ(sampler_config.rngs.size(), config.num_chains());
  EXPECT_EQ(sampler_config.init_params.size(), config.num_chains());
  
  // Check that init_params have the right size for each chain
  for (size_t i = 0; i < config.num_chains(); ++i) {
    EXPECT_EQ(sampler_config.init_params[i].size(), model_->num_params_r());
  }
}

TEST_F(LoadSamplersTest, LoadSamplers_UnitE_CorrectSizes) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(2)
    .seed(12345)
    .metric_type(stan::run::metric_t::UNIT_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
    
  auto sampler_config = stan::run::load_samplers<stan::run::metric_t::UNIT_E>(
    *model_, config, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(sampler_config.samplers.size(), config.num_chains());
  EXPECT_EQ(sampler_config.rngs.size(), config.num_chains());
  EXPECT_EQ(sampler_config.init_params.size(), config.num_chains());
  
  for (size_t i = 0; i < config.num_chains(); ++i) {
    EXPECT_EQ(sampler_config.init_params[i].size(), model_->num_params_r());
  }
}

TEST_F(LoadSamplersTest, LoadSamplers_SingleChain) {
  auto config = stan::run::hmc_nuts_config::create()
    .num_chains(1)
    .seed(12345)
    .metric_type(stan::run::metric_t::DIAG_E)
    .stepsize(1.0)
    .max_depth(10)
    .delta(0.8)
    .build();
  
  // Resize contexts for single chain
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  init_writers_.resize(1);
  
  auto sampler_config = stan::run::load_samplers<stan::run::metric_t::DIAG_E>(
    *model_, config, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(sampler_config.samplers.size(), 1);
  EXPECT_EQ(sampler_config.rngs.size(), 1);
  EXPECT_EQ(sampler_config.init_params.size(), 1);
  EXPECT_EQ(sampler_config.init_params[0].size(), model_->num_params_r());
}

TEST_F(LoadSamplersTest, SamplerTraits_TypeAliases) {
  // Test that sampler_traits compile correctly
  using diag_sampler = stan::run::sampler_traits<stan::run::metric_t::DIAG_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  using unit_sampler = stan::run::sampler_traits<stan::run::metric_t::UNIT_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  using dense_sampler = stan::run::sampler_traits<stan::run::metric_t::DENSE_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  
  // Just check that these types compile - we can't easily instantiate them here
  EXPECT_TRUE(true);  // Placeholder assertion
}
