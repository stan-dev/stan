#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <test/test-models/good/mcmc/hmc/hamiltonians/funnel.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <boost/random/additive_combine.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(BaseHamiltonian, update_potential_gradient) {

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  funnel_model_namespace::funnel_model model(data_var_context, &model_output);

  stan::mcmc::mock_hamiltonian<funnel_model_namespace::funnel_model, rng_t> metric(model);
  stan::mcmc::ps_point z(11);
  z.q.setOnes();

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  metric.update_potential_gradient(z, logger);

  EXPECT_FLOAT_EQ(10.73223197, z.V);

  EXPECT_FLOAT_EQ(8.757758279, z.g(0));
  for (int i = 1; i < z.q.size(); ++i)
    EXPECT_FLOAT_EQ(0.1353352832, z.g(i));


  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(BaseHamiltonian, streams) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  EXPECT_NO_THROW(funnel_model_namespace::funnel_model
                  model(data_var_context, 0,
                        static_cast<std::stringstream*>(0)));
  std::stringstream output;
  EXPECT_NO_THROW(funnel_model_namespace::funnel_model model(data_var_context, &output));
  EXPECT_EQ("", output.str());

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
