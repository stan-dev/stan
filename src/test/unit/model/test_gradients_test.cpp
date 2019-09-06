#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/model/test_gradients.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>


TEST(ModelUtil, streams) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  stan_model model(data_var_context, static_cast<std::stringstream*>(0));
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  std::stringstream out;

  try {
    stan::callbacks::stream_writer writer(out);
    stan::test::unit::instrumented_logger logger;
    out.str("");
    stan::model::test_gradients<true, true, stan_model>(model, params_r, params_i, 1e-6, 1e-6,
                                                        interrupt, logger, writer);
    EXPECT_EQ("\n Log probability=0\n\n param idx           value           model     finite diff           error\n         0               0               0               0               0\n", out.str());
    out.str("");
    stan::model::test_gradients<true, false, stan_model>(model, params_r, params_i, 1e-6, 1e-6,
                                                         interrupt, logger, writer);
    EXPECT_EQ("\n Log probability=0\n\n param idx           value           model     finite diff           error\n         0               0               0               0               0\n", out.str());
    out.str("");
    stan::model::test_gradients<false, true, stan_model>(model, params_r, params_i, 1e-6, 1e-6,
                                                         interrupt, logger, writer);
    EXPECT_EQ("\n Log probability=0\n\n param idx           value           model     finite diff           error\n         0               0               0               0               0\n", out.str());
    out.str("");
    stan::model::test_gradients<false, false, stan_model>(model, params_r, params_i, 1e-6, 1e-6,
                                                          interrupt, logger, writer);
    EXPECT_EQ("\n Log probability=0\n\n param idx           value           model     finite diff           error\n         0               0               0               0               0\n", out.str());
  } catch (...) {
    FAIL() << "test_gradients";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
