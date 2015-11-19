#ifndef STAN_MCMC_HMC_UNIFORM_BASE_STATIC_UNIFORM_HPP
#define STAN_MCMC_HMC_UNIFORM_BASE_STATIC_UNIFORM_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <boost/random/uniform_int_distribution.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo
    // with static integration time
    template <class M, class P, template<class, class> class H,
              template<class, class> class I, class BaseRNG>
    class base_static_uniform : public base_hmc<M, P, H, I, BaseRNG> {
    public:
      base_static_uniform(M &m, BaseRNG& rng, std::ostream* o, std::ostream* e)
        : base_hmc<M, P, H, I, BaseRNG>(m, rng, o, e), L_(1) {}

      ~base_static_uniform() {}

      sample transition(sample& init_sample) {
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_);

        ps_point z_init(this->z_);
        double H0 = this->hamiltonian_.H(this->z_);
        
        ps_point z_sample(this->z_);
        double sum_prob = 1;
        
        double sum_metro_prob = 0;
        
        boost::random::uniform_int_distribution<> uniform(0, L_ - 1);
        int Lp = uniform(this->rand_int_);

        for (int l = 0; l < Lp; ++l) {
          this->integrator_.evolve(this->z_,
                                   this->hamiltonian_,
                                   -this->epsilon_);

          double h = this->hamiltonian_.H(this->z_);
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          double prob = std::exp(H0 - h);
          sum_prob += prob;
          sum_metro_prob += prob > 1 ? 1 : prob;
          
          if (this->rand_uniform_() < prob / sum_prob)
            z_sample = this->z_;
        }
        
        this->z_.ps_point::operator=(z_init);
        
        for (int l = 0; l < L_ - 1 - Lp; ++l) {
          this->integrator_.evolve(this->z_,
                                   this->hamiltonian_,
                                   this->epsilon_);
          
          double h = this->hamiltonian_.H(this->z_);
          if (boost::math::isnan(h)) h = std::numeric_limits<double>::infinity();
          
          double prob = std::exp(H0 - h);
          sum_prob += prob;
          sum_metro_prob += prob > 1 ? 1 : prob;
          
          if (this->rand_uniform_() < prob / sum_prob)
            z_sample = this->z_;
        }
        
        double accept_prob = sum_metro_prob / static_cast<double>(L_ - 1);

        this->z_.ps_point::operator=(z_sample);
        return sample(this->z_.q, - this->hamiltonian_.V(this->z_), accept_prob);
      }

      void write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,L__,";
      }

      void write_sampler_params(std::ostream& o) {
        o << this->epsilon_ << "," << this->L_ << ",";
      }

      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("L__");
      }

      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->epsilon_);
        values.push_back(this->L_);
      }

      void set_L(int l) {
        if (l > 0) {
          L_ = l;
        }
      }

      void set_nominal_stepsize(const double e) {
        if (e > 0) {
          this->nom_epsilon_ = e;;
        }
      }

      int get_L() {
        return this->L_;
      }

    protected:
      int L_;

    };

  }  // mcmc
}  // stan
#endif
