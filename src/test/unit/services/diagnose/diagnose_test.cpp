#include <stan/services/diagnose/diagnose.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/stream_writer.hpp>

class ServicesDiagnose : public testing::Test {
public:
  ServicesDiagnose()
    : message(message_ss),
      init(init_ss),
      parameter(parameter_ss),
      model(context, &model_ss) {}

  std::stringstream message_ss, init_ss, parameter_ss, model_ss;
  stan::callbacks::stream_writer message, init, parameter;
  stan::io::empty_var_context context;
  stan::callbacks::interrupt interrupt;
  stan_model model;
};


TEST_F(ServicesDiagnose, diagnose) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;



  stan::services::diagnose::diagnose(model, context,
                                     seed, chain, init_radius,
                                     1e-6, 1e-6,
                                     interrupt,
                                     message, init, parameter);
  EXPECT_EQ("", model_ss.str());

  EXPECT_TRUE(message_ss.str().find("TEST GRADIENT MODE") != std::string::npos);
  EXPECT_TRUE(message_ss.str().find("Log probability=3.218") != std::string::npos);

  EXPECT_EQ("0,0\n", init_ss.str());

  EXPECT_TRUE(parameter_ss.str().find("Log probability=3.218") != std::string::npos);
}
