#ifndef STAN_SERVICES_UTIL_RUN_PARALLEL_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_PARALLEL_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/math/prim/mat.hpp>
#include <ctime>
#include <vector>
#include <iostream>
#include <memory>

namespace stan {
namespace services {
namespace util {

/**
 * Runs the sampler with adaptation.
 *
 * @tparam Sampler Type of adaptive sampler.
 * @tparam Model Type of model
 * @tparam RNG Type of random number generator
 * @param[in,out] sampler the mcmc sampler to use on the model
 * @param[in] model the model concept to use for computing log probability
 * @param[in] cont_vector initial parameter values
 * @param[in] num_warmup number of warmup draws
 * @param[in] num_samples number of post warmup draws
 * @param[in] num_thin number to thin the draws. Must be greater than
 *   or equal to 1.
 * @param[in] refresh controls output to the <code>logger</code>
 * @param[in] save_warmup indicates whether the warmup draws should be
 *   sent to the sample writer
 * @param[in,out] rng random number generator
 * @param[in,out] interrupt interrupt callback
 * @param[in,out] logger logger for messages
 * @param[in,out] sample_writer writer for draws
 * @param[in,out] diagnostic_writer writer for diagnostic information
 */
template <class Sampler, class Model, class RNG>
void run_parallel_adaptive_sampler(unsigned int num_chains,
                                   std::vector<std::unique_ptr<Sampler>>& sampler, Model& model,
                                   std::vector<std::vector<double>>& cont_vector,
                                   int num_warmup,
                                   int num_samples, int num_thin, int refresh,
                                   bool save_warmup, std::vector<std::unique_ptr<RNG>>& rng,
                                   callbacks::interrupt& interrupt,
                                   callbacks::logger& logger,
                                   callbacks::writer& sample_writer,
                                   callbacks::writer& diagnostic_writer) {

  std::cout << "running parallel adaptive for " << num_chains << " chains." << std::endl;
  std::vector<stan::mcmc::sample> s;
  //, stan::mcmc::sample(cont_params, 0, 0));

  std::vector<Eigen::VectorXd> cont_params;

  for(unsigned int i = 0; i != num_chains; ++i) {
    std::cout << "setting up chain " << i << std::endl;
    
    Eigen::Map<Eigen::VectorXd> cont_params_mapped(cont_vector[i].data(),
                                                   cont_vector[i].size());

    cont_params.push_back(cont_params_mapped);
    
    sampler[i]->engage_adaptation();
    try {
      sampler[i]->z().q = cont_params.back();
      sampler[i]->init_stepsize(logger);
    } catch (const std::exception& e) {
      logger.info("Exception initializing step size.");
      logger.info(e.what());
      return;
    }

    s.push_back(stan::mcmc::sample(cont_params.back(), 0, 0));
  }

  services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);

  // Headers
  writer.write_sample_names(s[0], *sampler[0], model);
  writer.write_diagnostic_names(s[0], *sampler[0], model);

  Eigen::MatrixXd var_buffer(model.num_params_r(), num_chains);

  clock_t start = clock();
  std::cout << "Running warmup for " << num_warmup << " steps" << std::endl;

  //std::vector<bool> 
  
  // run all samplers in turn
  for(int i=0; i != num_warmup; ++i) {
    bool pool_metric = false;
    for(unsigned int j=0; j != num_chains; ++j) {
      Sampler& active_sampler = *sampler[j];
      const bool end_adaptation_window = active_sampler.get_var_adaptation().end_adaptation_window();
      
      if(active_sampler.get_var_adaptation().end_adaptation_window()) {
        std::cout << "iteration " << i << " ends an adaptation window" << " for thread " << j << std::endl;
      }

      util::generate_transitions(active_sampler, 1, i, num_warmup + num_samples,
                                 num_thin, refresh, save_warmup, true, writer, s[j],
                                 model, *rng[j], interrupt, logger);

      if(end_adaptation_window) {
        var_buffer.col(j) = active_sampler.z().inv_e_metric_;
        pool_metric = true;
      }
    }
    if(pool_metric) {
      // TODO: pool stepsize
      std::cout << "POOLING metric." << std::endl;
      Eigen::VectorXd pooled_inv_metric = var_buffer.rowwise().mean();
      for(std::size_t j=0; j != num_chains; ++j) {
        sampler[j]->set_metric(pooled_inv_metric);
      }
    }
  }
  clock_t end = clock();
  double warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

  for(int j=0; j != num_chains; ++j) {
    sampler[j]->disengage_adaptation();
    writer.write_adapt_finish(*sampler[j]);
    sampler[j]->write_sampler_state(sample_writer);
  }
  // seems to confuse the writers otherwise => we can only save warmup
  // info of the first chain at the moment
  //writer.write_adapt_finish(sampler[0]);
  //sampler[0].write_sampler_state(sample_writer);

  start = clock();
  std::cout << "Running sampling for " << num_samples << " steps" << std::endl;
  for(int i=0; i != num_samples; ++i) {
    for(unsigned int j=0; j != num_chains; ++j) {
      util::generate_transitions(*sampler[j], 1, num_warmup+i,
                                 num_warmup + num_samples, num_thin, refresh, true,
                                 false, writer, s[j], model, *rng[j], interrupt, logger);
    }
  }
  
  end = clock();
  double sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;

  writer.write_timing(warm_delta_t, sample_delta_t);
}
}  // namespace util
}  // namespace services
}  // namespace stan
#endif
