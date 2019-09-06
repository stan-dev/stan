#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts_classic/unit_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/diag_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/dense_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/adapt_unit_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/adapt_diag_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/adapt_dense_e_nuts_classic.hpp>
#include <boost/random/additive_combine.hpp>
#include <stan/io/dump.hpp>
#include <fstream>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcNutsClassic, instantiaton_test) {
  rng_t base_rng(4839294);

  std::stringstream output;
  stan::callbacks::stream_writer writer(output);
  std::stringstream error_stream;
  stan::callbacks::stream_writer error_writer(error_stream);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::unit_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    unit_e_sampler(model, base_rng);
  
  stan::mcmc::diag_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    diag_e_sampler(model, base_rng);
  
  stan::mcmc::dense_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    dense_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_unit_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_unit_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_diag_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_diag_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_dense_e_nuts_classic<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_dense_e_sampler(model, base_rng);
}
