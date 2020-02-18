#ifndef STAN_MCMC_HMC_NUTS_ADAPT_AUTO_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_ADAPT_AUTO_E_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/nuts/auto_e_nuts.hpp>
#include <stan/mcmc/hmc/mpi_cross_chain_adapter.hpp>
#include <stan/mcmc/mpi_auto_adaptation.hpp>

namespace stan {
namespace mcmc {
/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and adaptive
 * dense or diagonal metric and adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_auto_e_nuts : public auto_e_nuts<Model, BaseRNG>,
			  public mpi_cross_chain_adapter<adapt_auto_e_nuts<Model, BaseRNG>>,
			  public stepsize_covar_adapter {
protected:
  const Model& model_;
public:
  adapt_auto_e_nuts(const Model& model, BaseRNG& rng)
    : model_(model), auto_e_nuts<Model, BaseRNG>(model, rng),
    stepsize_covar_adapter(model.num_params_r()) {}

  ~adapt_auto_e_nuts() {}

  sample
  transition(sample& init_sample, callbacks::logger& logger) {
    sample s = auto_e_nuts<Model, BaseRNG>::transition(init_sample,
						       logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
						s.accept_stat());

      bool update;
      if (this -> use_cross_chain_adapt()) {
	this -> add_cross_chain_sample(s.log_prob());
	update = this -> cross_chain_adaptation(logger);
	if (this -> is_cross_chain_adapted()) {
	  update = false;
	}
      } else {
	update = this->covar_adaptation_.learn_covariance(this->z_.inv_e_metric_,
							  this->z_.q);
      }

      if (update) {
	//std::cout << this->z_.inv_e_metric_ << std::endl;
	this->z_.is_diagonal_ = reinterpret_cast<mpi_auto_adaptation<Model> *>(this->var_adapt)->is_diagonal_;

	this->init_stepsize(logger);

	this->stepsize_adaptation_.set_mu(log(10 * this->nom_epsilon_));
	this->stepsize_adaptation_.restart();

	if (this -> use_cross_chain_adapt()) {
	  this->set_cross_chain_stepsize();          
	}
      }
    }
    return s;
  }

  void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    if (!this -> is_cross_chain_adapted()) {
      this->stepsize_adaptation_.complete_adaptation(this->nom_epsilon_);
    }
  }
};

}  // mcmc
}  // stan
#endif
