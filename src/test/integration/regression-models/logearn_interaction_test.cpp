#include <gtest/gtest.h>
#include <ctime>
#include <test/test-models/regression/logearn_interaction.hpp>
#include <test/performance/utility.hpp>
#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG


typedef boost::ecuyer1988 rng_t;

class logearn_interaction : public ::testing::Test {
public:
  logearn_interaction()
    : rng0(0), rng1(1000), rng2(2000), rng3(3000),
      data_stream("src/test/test-models/regression/logearn_interaction.data.R"),
      data(data_stream),
      model(data, &std::cout),
      cont_params(Eigen::VectorXd::Zero(model.num_params_r())),
      s(cont_params, 0, 0) {
  }
  
  void SetUp() {
    num_warmup = 1000;
    num_samples = 1000;
  }

  rng_t rng0, rng1, rng2, rng3;
  std::ifstream data_stream;
  stan::io::dump data;
  stan_model model;
  int num_warmup, num_samples;
  Eigen::VectorXd cont_params;
  stan::mcmc::sample s;
  stan::interface_callbacks::interrupt::noop noop;
};

TEST_F(logearn_interaction, nuts) {
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t>
    sampler(model, rng0, &std::cout, &std::cout);

  stan::interface_callbacks::writer::stream_writer
    sample(std::cout, "# "),
    diagnostic(std::cout, "# "),
    message(std::cout, " ");
  
  stan::services::sample::mcmc_writer<stan_model,
                                      stan::interface_callbacks::writer::stream_writer,
                                      stan::interface_callbacks::writer::stream_writer,
                                      stan::interface_callbacks::writer::stream_writer>
    writer(sample, diagnostic, message, &std::cout);
  
  stan::services::mcmc::warmup(&sampler, num_warmup, num_samples, 1,
                               1000, true,
                               writer,
                               s, model, rng0,
                               "", "", std::cout,
                               noop);
  
  sampler.disengage_adaptation();

  stan::services::mcmc::sample(&sampler, num_warmup, num_samples, 1,
                               1000, true,
                               writer, 
                               s, model, rng0,
                               "", "", std::cout,
                               noop);
}
