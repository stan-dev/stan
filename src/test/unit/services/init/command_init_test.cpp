#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <boost/random/additive_combine.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/services/sample/init_adapt.hpp>

typedef test_lp_model_namespace::test_lp_model Model;
typedef boost::ecuyer1988 rng_t;
typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;

class UiCommand : public testing::Test {
public:
  UiCommand()
    : writer(output),
      error_writer(error) {}
  
  void SetUp() {
    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model_output.str("");
    model = new Model(empty_data_context, &model_output);
    base_rng.seed(123456);
    
    output.str("");
    error.str("");
    sampler_ptr = new sampler((*model), base_rng);
    sampler_ptr->set_nominal_stepsize(1);
    sampler_ptr->set_stepsize_jitter(0);
    sampler_ptr->set_max_depth(10);

    delta = 0.8;
    gamma = 0.05;
    kappa = 0.75;
    t0 = 10;
    
    z_0.resize(model->num_params_r());
    z_0.fill(0);
    z_init.resize(model->num_params_r());
    z_init.fill(1);
  }
  
  void TearDown() {
    delete sampler_ptr;
    delete model;
  }
  
  rng_t base_rng;
  Model* model;
  sampler* sampler_ptr;
  double delta, gamma, kappa, t0;
  Eigen::VectorXd z_0;
  Eigen::VectorXd z_init;

  std::stringstream model_output, output, error;
  stan::interface_callbacks::writer::stream_writer writer;
  stan::interface_callbacks::writer::stream_writer error_writer;
};

TEST_F(UiCommand, init_adapt_z_0) {
  EXPECT_TRUE(stan::services::sample::init_adapt(sampler_ptr,
                                                 delta, gamma, kappa, t0,
                                                 z_0, writer, error_writer));
  EXPECT_FLOAT_EQ(0.125, sampler_ptr->get_nominal_stepsize());

  for (size_t n = 0; n < model->num_params_r(); n++) {
    EXPECT_FLOAT_EQ(z_0[n], sampler_ptr->z().q[n]);
  }

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}


TEST_F(UiCommand, init_adapt_z_init) {
  EXPECT_TRUE(stan::services::sample::init_adapt(sampler_ptr,
                                                 delta, gamma, kappa, t0,
                                                 z_init, writer, error_writer));
  EXPECT_FLOAT_EQ(0.25, sampler_ptr->get_nominal_stepsize());
  for (size_t n = 0; n < model->num_params_r(); n++) {
    EXPECT_FLOAT_EQ(z_init[n], sampler_ptr->z().q[n]);
  }

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());
}
