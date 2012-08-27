#ifndef __STAN__MCMC__ADAPTIVE_HMC_H__
#define __STAN__MCMC__ADAPTIVE_HMC_H__

#include <ctime>
#include <cstddef>
#include <iostream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/math/util.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/dualaverage.hpp>
#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/util.hpp>
#include <stan/model/prob_grad.hpp>

namespace stan {

  namespace mcmc {

    /**
     * Adaptive Hamiltonian Monte Carlo (HMC) sampler. 
     *
     * adaptive_hmc automatically adapts the step size, epsilon,
     * to try to coerce the average acceptance probability to
     * some value, delta.
     *
     * The HMC sampler requires a probability model with the ability
     * to compute gradients.  This is provided through an instance of
     * <code>stan::model::prob_grad</code>.  
     *
     * Samples from the sampler are returned through the superclass
     * <code>stan::mcmc::sample</code> and adaptation is handled
     * through the superclass <code>stan::mcmc::adaptive_sampler</code>.
     */
    template <class BaseRNG = boost::mt19937>
    class adaptive_hmc : public hmc_base<BaseRNG> {
    private:
    
      unsigned int _L;   // fixed number of Hamiltonian simulation steps

    public:

      /**
       * Construct an adaptive Hamiltonian Monte Carlo (HMC) sampler
       * for the specified model, using the specified step size and
       * number of leapfrog steps, with the specified random seed for
       * randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param L Number of leapfrog steps per simulation.
       * @param delta Target value of E[acceptance probability]. Optional;
       * defaults to the value of 0.651, which has some theoretical
       * justification.
       * @param epsilon Hamiltonian dynamics simulation step size. Optional;
       * if not specified or set < 0, find_reasonable_parameters() will be 
       * called to initialize epsilon.
       * @param epsilon_pm
       * @param epsilon_adapt
       * @param delta Target value of E[acceptance probability]. 
       * Optional; defaults to the value of 0.651, which has some 
       * theoretical justification.
       * @param gamma Regularization parameter. See 
       * <code>stan::mcmc::DualAverage</code>.
       * @param rand_int Seed for random number generator; optional, if not
       * specified, generate new seen based on system time.
       */
      adaptive_hmc(stan::model::prob_grad& model,
                   int L, 
                   double epsilon=-1,
                   double epsilon_pm = 0.0,
                   bool epsilon_adapt = true, 
                   double delta = 0.651,
                   double gamma = 0.05,
                   BaseRNG rand_int = BaseRNG(std::time(0)))
        : hmc_base<BaseRNG>(model,
                            epsilon,
                            epsilon_pm,
                            epsilon_adapt,
                            delta,
                            gamma,
                            rand_int),
          _L(L)
      {
        this->adaptation_init(1.0);  // target is just epsilon
      }

      /**
       * Destructor. The implementation for this class is a no-op.
       */
      ~adaptive_hmc() {
      }

      




      /**
       * Returns the next sample.
       *
       * @return The next sample.
       */
      virtual sample next_impl() {
        // Gibbs for discrete
        // std::vector<double> probs;
        // for (size_t m = 0; m < this->_model.num_params_i(); ++m) {
        //   probs.resize(0);
        //   for (int k = this->_model.param_range_i_lower(m); 
        //        k < this->_model.param_range_i_upper(m); 
        //        ++k)
        //     probs.push_back(this->_model.log_prob_star(m,k,this->_x,this->_z));
        //   this->_z[m] = sample_unnorm_log(probs,this->_rand_uniform_01);
        // }
        // HMC for continuous
        std::vector<double> m(this->_model.num_params_r());
        for (size_t i = 0; i < m.size(); ++i)
          m[i] = this->_rand_unit_norm();
        double H = -(stan::math::dot_self(m) / 2.0) + this->_logp; 
        
        std::vector<double> g_new(this->_g);
        std::vector<double> x_new(this->_x);
        double logp_new = -1e100;
        double epsilon = this->_epsilon;
        // only vary epsilon after done adapting
        if (!this->adapting() && this->varying_epsilon()) { 
          double low = epsilon * (1.0 - this->_epsilon_pm);
          double high = epsilon * (1.0 + this->_epsilon_pm);
          double range = high - low;
          epsilon = low + (range * this->_rand_uniform_01());
        }
        this->_epsilon_last = epsilon;
        for (unsigned int l = 0; l < _L; ++l)
          logp_new = leapfrog(this->_model, this->_z, x_new, m, g_new, epsilon,
                              this->_error_msgs);
        this->nfevals_plus_eq(_L);

        double H_new = -(stan::math::dot_self(m) / 2.0) + logp_new;
        double dH = H_new - H;
        if (this->_rand_uniform_01() < exp(dH)) {
          this->_x = x_new;
          this->_g = g_new;
          this->_logp = logp_new;
        }

        // Now we just have to update epsilon, if adaptation is on.
        double adapt_stat = stan::math::min(1, exp(dH));
        if (adapt_stat != adapt_stat)
          adapt_stat = 0;
        if (this->adapting()) {
          double adapt_g = adapt_stat - this->_delta;
          std::vector<double> gvec(1, -adapt_g);
          std::vector<double> result; // FIXME: update directly to _epsilon?
          this->_da.update(gvec, result);
          this->_epsilon = exp(result[0]);
        }
        std::vector<double> result;
        this->_da.xbar(result);
        // fprintf(stderr, "xbar = %f\n", exp(result[0]));
        double avg_eta = 1.0 / this->n_steps();
        this->update_mean_stat(avg_eta,adapt_stat);

        return mcmc::sample(this->_x, this->_z, this->_logp);
      }

      virtual void write_sampler_param_names(std::ostream& o) {
        if (this->_epsilon_adapt || this->varying_epsilon())
          o << "stepsize__,";
      }

      virtual void write_sampler_params(std::ostream& o) {
        if (this->_epsilon_adapt || this->varying_epsilon())
          o << this->_epsilon_last << ',';
      }

      virtual void get_sampler_param_names(std::vector<std::string>& names) {
        names.clear();
        if (this->_epsilon_adapt || this->varying_epsilon())
          names.push_back("stepsize__");
      }

      virtual void get_sampler_params(std::vector<double>& values) {
        values.clear();
        if (this->_epsilon_adapt || this->varying_epsilon())
          values.push_back(this->_epsilon_last);
      }
    };

  }

}

#endif
