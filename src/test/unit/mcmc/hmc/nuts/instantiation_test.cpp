#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/empty_var_context.hpp>
#include <fstream>

#include <gtest/gtest.h>

TEST(McmcNuts, instantiaton_test) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  std::stringstream output;
  stan::callbacks::stream_writer writer(output);
  std::stringstream error_stream;
  stan::callbacks::stream_writer error_writer(error_stream);

  stan::io::empty_var_context data_var_context;
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::unit_e_nuts<gauss3D_model_namespace::gauss3D_model, stan::rng_t>
      unit_e_sampler(model, base_rng);

  stan::mcmc::diag_e_nuts<gauss3D_model_namespace::gauss3D_model, stan::rng_t>
      diag_e_sampler(model, base_rng);

  stan::mcmc::dense_e_nuts<gauss3D_model_namespace::gauss3D_model, stan::rng_t>
      dense_e_sampler(model, base_rng);

  stan::mcmc::adapt_unit_e_nuts<gauss3D_model_namespace::gauss3D_model,
                                stan::rng_t>
      adapt_unit_e_sampler(model, base_rng);

  stan::mcmc::adapt_diag_e_nuts<gauss3D_model_namespace::gauss3D_model,
                                stan::rng_t>
      adapt_diag_e_sampler(model, base_rng);

  stan::mcmc::adapt_dense_e_nuts<gauss3D_model_namespace::gauss3D_model,
                                 stan::rng_t>
      adapt_dense_e_sampler(model, base_rng);
}
