#ifndef __STAN__MCMC__HMC_HPP__
#define __STAN__MCMC__HMC_HPP__

#include <ctime>
#include <cstddef>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/sampler.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/util.hpp>
#include <stan/math/util.hpp>

namespace stan {

  namespace mcmc {


    /**
     * Hamiltonian Monte Carlo sampler.
     *
     * The HMC sampler requires a probability model with the ability
     * to compute gradients, characterized as an instance of
     * <code>prob_grad</code>.  
     *
     * Samples from the sampler are returned through the
     * base class <code>sampler</code>.
     */
    class hmc : public adaptive_sampler {
    private:
      mcmc::prob_grad& _model;
    
      std::vector<double> _x;
      std::vector<int> _z;
      std::vector<double> _g;
      double _logp;

      double _epsilon;
      unsigned int _L;

      boost::mt19937 _rand_int;
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > _rand_unit_norm;
      boost::uniform_01<boost::mt19937&> _rand_uniform_01;

    public:

      /**
       * Construct a Hamiltonian Monte Carlo (HMC) sampler for the
       * specified model, using the specified step size and number of
       * leapfrog steps, with the specified random seed for randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param epsilon Hamiltonian dynamics simulation step size.
       * @param L Number of leapfrog steps per simulation.
       * @param random_seed Seed for random number generator; optional, if not
       * specified, generate new seen based on system time.
       */
      hmc(mcmc::prob_grad& model,
          double epsilon, 
          int L,
          unsigned int random_seed = static_cast<unsigned int>(std::time(0)))
        : adaptive_sampler(false),
          _model(model),
          _x(model.num_params_r()),
          _z(model.num_params_i()),
          _g(model.num_params_r()),

          _epsilon(epsilon),
          _L(L),

          _rand_int(random_seed),
          _rand_unit_norm(_rand_int,
                          boost::normal_distribution<>()),
          _rand_uniform_01(_rand_int) {

        model.init(_x,_z);
        _logp = model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Destroy this sampler.
       *
       * The implementation for this class is a no-op.
       */
      virtual ~hmc() {
      }

      /**
       * Set the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       * @throw std::invalid_argument if x or z do not match size 
       *    of parameters specified by the model.
       */
      virtual void set_params(const std::vector<double>& x, 
                              const std::vector<int>& z) {
        if (x.size() != _x.size() || z.size() != _z.size())
          throw std::invalid_argument("x.size() or z.size() mismatch");
        _x = x;
        _z = z;
      }

      /**
       * Set the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @throw std::invalid_argument if the number of real parameters does
       *   not match the number of parameters defined by the model.
       */
      virtual void set_params_r(const std::vector<double>& x) {
        if (x.size() != _model.num_params_r())
          throw std::invalid_argument ("x.size() must match the number of parameters of the model.");
        _x = x;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Set the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param z Integer parameters.
       * @throw std::invalid_argument if the number of integer parameters does
       *   not match the number of parameters defined by the model.
       */
      virtual void set_params_i(const std::vector<int>& z) {
        if (z.size() != _model.num_params_i())
          throw std::invalid_argument ("z.size() must match the number of parameters of the model.");
        _z = z;
        _logp = _model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      virtual sample next_impl() {
        // Gibbs for discrete
        std::vector<double> probs;
        for (size_t m = 0; m < _model.num_params_i(); ++m) {
          probs.resize(0);
          for (int k = _model.param_range_i_lower(m); 
               k < _model.param_range_i_upper(m); 
               ++k)
            probs.push_back(_model.log_prob_star(m,k,_x,_z));
          _z[m] = sample_unnorm_log(probs,_rand_uniform_01);
        }

        // HMC for continuous
        std::vector<double> m(_model.num_params_r());
        for (size_t i = 0; i < m.size(); ++i)
          m[i] = _rand_unit_norm();
        double H = -(stan::math::dot_self(m) / 2.0) + _logp; 
        
        std::vector<double> g_new(_g);
        std::vector<double> x_new(_x);
        //double epsilon_over_2 = _epsilon / 2.0;

        double logp_new = -1e100;
        for (unsigned int l = 0; l < _L; ++l)
          logp_new = leapfrog(_model, _z, x_new, m, g_new, _epsilon);
        nfevals_plus_eq(_L);

        double H_new = -(stan::math::dot_self(m) / 2.0) + logp_new;
        double dH = H_new - H;
        if (_rand_uniform_01() < exp(dH)) {
          _x = x_new;
          _g = g_new;
          _logp = logp_new;
        }
        mcmc::sample s(_x,_z,_logp);
        return s;
      }

    };

  }

}

#endif
