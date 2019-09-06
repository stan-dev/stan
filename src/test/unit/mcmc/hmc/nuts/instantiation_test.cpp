#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <boost/random/additive_combine.hpp>
#include <stan/io/dump.hpp>
#include <fstream>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcNuts, instantiaton_test) {
  rng_t base_rng(4839294);

  std::stringstream output;
  stan::callbacks::stream_writer writer(output);
  std::stringstream error_stream;
  stan::callbacks::stream_writer error_writer(error_stream);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::unit_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    unit_e_sampler(model, base_rng);
  
  stan::mcmc::diag_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    diag_e_sampler(model, base_rng);
  
  stan::mcmc::dense_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    dense_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_unit_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_unit_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_diag_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_diag_e_sampler(model, base_rng);
  
  stan::mcmc::adapt_dense_e_nuts<gauss3D_model_namespace::gauss3D_model, rng_t>
    adapt_dense_e_sampler(model, base_rng);
}
