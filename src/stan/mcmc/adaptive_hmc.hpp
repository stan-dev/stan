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
     * to compute gradients, characterized as an instance of
     * <code>stan::model::prob_grad</code>.  
     *
     * Samples from the sampler are returned through the
     * base class <code>stan::mcmc::sample</code>.
     */
    template <class BaseRNG = boost::mt19937>
    class adaptive_hmc : public hmc_base<BaseRNG> {
    private:

      // FIXME:  make _model constant
      // Provides the target distribution we're trying to sample from
      stan::model::prob_grad& _model;
    
      // The number of steps used in the Hamiltonian simulation
      unsigned int _L;

      // The step size used in the Hamiltonian simulation
      double _epsilon;
      // The +/- around epsilon
      double _epsilon_pm;

      // last value of epsilon used by the algorithm
      double _last_epsilon;
      bool _adapt_epsilon;


      // The desired value of E[acceptance probability]
      const double _delta;

      // The most recent setting of the real-valued parameters
      std::vector<double> _x;
      // The most recent setting of the discrete parameters
      std::vector<int> _z;
      // The most recent gradient with respect to the real parameters
      std::vector<double> _g;
      // The most recent log-likelihood
      double _logp;

      // Class implementing Nesterov's primal-dual averaging
      DualAverage _da;

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
       * @param base_rng Seed for random number generator; optional, if not
       * specified, generate new seen based on system time.
       */
      adaptive_hmc(stan::model::prob_grad& model,
                   int L, 
                   double epsilon=-1,
                   double epsilon_pm = 0.0,
                   bool epsilon_adapt = true, 
                   double delta = 0.651,
                   double gamma = 0.05,
                   BaseRNG base_rng = BaseRNG(std::time(0)))
        : hmc_base<BaseRNG>(epsilon_adapt,base_rng),

          _model(model),

          _L(L),

          _epsilon(epsilon),
          _epsilon_pm(epsilon_pm),
          _last_epsilon(epsilon),
          _adapt_epsilon(epsilon < 0.0),

          _delta(delta),
          
          _x(model.num_params_r()),
          _z(model.num_params_i()),
          _g(model.num_params_r()),

          _da(gamma, std::vector<double>(1, 0)) {
        model.init(_x,_z);
        _logp = model.grad_log_prob(_x,_z,_g);
        if (_adapt_epsilon)
          find_reasonable_parameters();
        _da.setx0(std::vector<double>(1, log(_epsilon)));
      }

      /**
       * Destructor. The implementation for this class is a no-op.
       */
      virtual ~adaptive_hmc() {
      }

      /**
       * Sets the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       */
      virtual void set_params(const std::vector<double>& x,
                              const std::vector<int>& z) {
        assert(x.size() == _x.size());
        assert(z.size() == _z.size());
        _x = x;
        _z = z;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Sets the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @throw std::invalid_argument if the number of real parameters does
       *   not match the number of parameters defined by the model.
       */
      void set_params_r(const std::vector<double>& x) {
        if (x.size() != _model.num_params_r())
          throw std::invalid_argument("x.size() must match number of model params.");
        _x = x;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Sets the model integer parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param z Integer parameters.
       * @throw std::invalid_argument if the number of integer parameters does
       *   not match the number of parameters defined by the model.
       */
      void set_params_i(const std::vector<int>& z) {
        if (z.size() != _model.num_params_i())
          throw std::invalid_argument ("z.size() must match the number of parameters of the model.");
        _z = z;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Search for a roughly reasonable (within a factor of 2)
       * setting of the step size epsilon.
       */
      virtual void find_reasonable_parameters() {
        _epsilon = 1.0;
        std::vector<double> x = _x;
        std::vector<double> m(_model.num_params_r());
        for (size_t i = 0; i < m.size(); ++i)
          m[i] = this->_rand_unit_norm();
        std::vector<double> g = _g;
        double lastlogp = _logp;
        double logp = leapfrog(_model, _z, x, m, g, _epsilon);
        double H = logp - lastlogp;
        int direction = H > log(0.5) ? 1 : -1;
        // fprintf(stderr, "epsilon = %f.  initial logp = %f, lf logp = %f\n", 
        //   _epsilon, lastlogp, logp);
        while (1) {
          x = _x;
          g = _g;
          for (size_t i = 0; i < m.size(); ++i)
            m[i] = this->_rand_unit_norm();
          logp = leapfrog(_model, _z, x, m, g, _epsilon);
          H = logp - lastlogp;
          // fprintf(stderr, "epsilon = %f.  initial logp = %f, lf logp = %f\n", 
          //   _epsilon, lastlogp, logp);
          if ((direction == 1) && (H < log(0.5)))
            break;
          else if ((direction == -1) && (H > log(0.5)))
            break;
          else
            _epsilon = (direction == 1) ? 2.0 * _epsilon : 0.5 * _epsilon;
        }
      }

      bool varying_epsilon() {
        return _epsilon_pm != 0;
      }

      /**
       * Returns the next sample.
       *
       * @return The next sample.
       */
      virtual sample next_impl() {
        // Gibbs for discrete
        // std::vector<double> probs;
        // for (size_t m = 0; m < _model.num_params_i(); ++m) {
        //   probs.resize(0);
        //   for (int k = _model.param_range_i_lower(m); 
        //        k < _model.param_range_i_upper(m); 
        //        ++k)
        //     probs.push_back(_model.log_prob_star(m,k,_x,_z));
        //   _z[m] = sample_unnorm_log(probs,this->_rand_uniform_01);
        // }
        // HMC for continuous
        std::vector<double> m(_model.num_params_r());
        for (size_t i = 0; i < m.size(); ++i)
          m[i] = this->_rand_unit_norm();
        double H = -(stan::math::dot_self(m) / 2.0) + _logp; 
        
        std::vector<double> g_new(_g);
        std::vector<double> x_new(_x);
        double logp_new = -1e100;
        double epsilon = _epsilon;
        // only vary epsilon after done adapting
        if (!this->adapting() && varying_epsilon()) { 
          double low = epsilon * (1.0 - _epsilon_pm);
          double high = epsilon * (1.0 + _epsilon_pm);
          double range = high - low;
          epsilon = low + (range * this->_rand_uniform_01());
        }
        _last_epsilon = epsilon;
        for (unsigned int l = 0; l < _L; ++l)
          logp_new = leapfrog(_model, _z, x_new, m, g_new, epsilon);
        this->nfevals_plus_eq(_L);

        double H_new = -(stan::math::dot_self(m) / 2.0) + logp_new;
        double dH = H_new - H;
        if (this->_rand_uniform_01() < exp(dH)) {
          _x = x_new;
          _g = g_new;
          _logp = logp_new;
        }

        // Now we just have to update epsilon, if adaptation is on.
        double adapt_stat = stan::math::min(1, exp(dH));
        if (adapt_stat != adapt_stat)
          adapt_stat = 0;
        if (this->adapting()) {
          double adapt_g = adapt_stat - _delta;
          std::vector<double> gvec(1, -adapt_g);
          std::vector<double> result; // FIXME: update directly to _epsilon?
          _da.update(gvec, result);
          _epsilon = exp(result[0]);
        }
        std::vector<double> result;
        _da.xbar(result);
        // fprintf(stderr, "xbar = %f\n", exp(result[0]));
        double avg_eta = 1.0 / this->n_steps();
        this->update_mean_stat(avg_eta,adapt_stat);

        return mcmc::sample(_x, _z, _logp);
      }

      /**
       * Turn off parameter adaptation. Because we're using
       * primal-dual averaging, once we're done adapting we want to
       * set epsilon=the _average_ value of epsilon over each
       * adaptation step. This results in a lower-variance estimate of
       * the optimal epsilon.
       */
      virtual void adapt_off() {
        if (!this->adapting()) return;
        adaptive_sampler::adapt_off();
        std::vector<double> result;
        _da.xbar(result);
        _epsilon = exp(result[0]);
      }

      /**
       * Returns the value of epsilon.
       *
       * @param[out] params Where to store epsilon.
       */
      virtual void get_parameters(std::vector<double>& params) {
        params.assign(1, _epsilon);
      }


      virtual void write_sampler_param_names(std::ostream& o) {
        if (_adapt_epsilon || varying_epsilon())
          o << "stepsize__,";
      }

      virtual void write_sampler_params(std::ostream& o) {
        if (_adapt_epsilon || varying_epsilon())
          o << _last_epsilon << ',';
      }


    };

  }

}

#endif
