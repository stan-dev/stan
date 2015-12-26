#include <stan/services/diagnose/diagnose.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>

class ServicesDiagnose : public testing::Test {
public:
  ServicesDiagnose()
    : message(message_ss), parameter(parameter_ss) {}

  void SetUp() {
    message_ss.str("");
    parameter_ss.str("");
    model_ss.str("");

    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();

    model = new stan_model(empty_data_context, &model_ss);

    cont_params = Eigen::VectorXd::Zero(model->num_params_r());
  }

  void TearDown() {
    delete model;
  }
  
  std::stringstream message_ss, parameter_ss, model_ss;
  stan::interface_callbacks::writer::stream_writer message, parameter;
  Eigen::VectorXd cont_params;
  stan_model* model;
};


TEST_F(ServicesDiagnose, diagnose) {
  stan::services::diagnose::diagnose(cont_params, *model, 1e-6, 1e-6,
    message, parameter);

  EXPECT_EQ("", model_ss.str());

  EXPECT_TRUE(message_ss.str().find("TEST GRADIENT MODE") != std::string::npos);
  EXPECT_TRUE(message_ss.str().find("Log probability=3.218") != std::string::npos);

  EXPECT_TRUE(parameter_ss.str().find("TEST GRADIENT MODE") == std::string::npos);
  EXPECT_TRUE(parameter_ss.str().find("Log probability=3.218") != std::string::npos);
}
