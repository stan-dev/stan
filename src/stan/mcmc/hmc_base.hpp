#ifndef __STAN__MCMC__HMC_BASE_H__
#define __STAN__MCMC__HMC_BASE_H__

#include <ctime>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/dualaverage.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/util.hpp>


namespace stan {

  namespace mcmc {

    template <class BaseRNG = boost::mt19937>
    class hmc_base : public adaptive_sampler {

    protected:

      // model from which to sample
      stan::model::prob_grad& _model;   // model to sample

      double _epsilon;                  // step size for Hamiltonian sim
      double _epsilon_pm;               // +/- around epsilon
      double _epsilon_last;             // last value of epsilon used
      bool _epsilon_adapt;              // true if adapt(ed) epsilon

      const double _delta;              // target E[accept]
      const double _gamma;              // tuning param for dual avg
      DualAverage _da;                  // impl of dual avg adaptation

      BaseRNG _rand_int;                // base random number generator

      boost::variate_generator<BaseRNG&, 
                               boost::normal_distribution<> > _rand_unit_norm;
                                        // normal(0,1) RNG

      boost::uniform_01<BaseRNG&> _rand_uniform_01;                
                                        // uniform(0,1) RNG

      std::vector<double> _x;           // most recent real params
      std::vector<int> _z;              // most recent discrete params
      std::vector<double> _g;           // most recent gradient
      double _logp;                     // most recent log prob

      /**
       * Search for a roughly reasonable (within a factor of 2)
       * setting of the step size epsilon.
       */
      virtual void find_reasonable_parameters() {
        this->_epsilon = 1.0;
        std::vector<double> x = this->_x;
        std::vector<double> m(this->_model.num_params_r());
        for (size_t i = 0; i < m.size(); ++i)
          m[i] = this->_rand_unit_norm();
        std::vector<double> g = this->_g;
        double lastlogp = this->_logp;
        double logp = leapfrog(this->_model, this->_z, x, m, g, this->_epsilon);
        double H = logp - lastlogp;
        int direction = H > log(0.5) ? 1 : -1;
        while (1) {
          x = this->_x;
          g = this->_g;
          for (size_t i = 0; i < m.size(); ++i)
            m[i] = this->_rand_unit_norm();
          logp = leapfrog(this->_model, this->_z, x, m, g, this->_epsilon);
          H = logp - lastlogp;
          if ((direction == 1) && !(H > log(0.5)))
            break;
          else if ((direction == -1) && !(H < log(0.5)))
            break;
          else
            this->_epsilon = ( (direction == 1) 
                               ? 2.0 * this->_epsilon 
                               : 0.5 * this->_epsilon );

	  if (this->_epsilon > 1e300)
	    throw std::runtime_error("Posterior is improper. Please check your model.");
	  if (this->_epsilon == 0)
	    throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
        }
      }

      void adaptation_init(double epsilon_scale) {
        if (this->adapting())
          this->_da.setx0(std::vector<double>(1, log(epsilon_scale * _epsilon)));
      }

    public:

      /**
       * Construct a base HMC sampler.
       *
       * @param model Log probability model with gradients.
       * @param epsilon Hamiltonian dynamics simulation step size. Optional;
       * if not specified or set < 0, find_reasonable_parameters() will be 
       * called to initialize epsilon; default value = -1
       * @param epsilon_pm Sample in range defined by plus-or minus
       * this value over epsilon (sample step size uniformly in interval
       * <code>[epsilon*(1-epsilon_pm),
       * epsilon*(1+epsilon_pm)]</code>;
       * default value = 0.0
       * @param epsilon_adapt Flag indicating whether adaptation is
       * turned on.
       * @param delta Target acceptance rate for adaptation.
       * @param gamma Tuning parameter for dual averaging adaptation.
       * @param rand_int Base random integer generator.
       */
      hmc_base(stan::model::prob_grad& model,
               double epsilon=-1,
               double epsilon_pm = 0.0,
               bool epsilon_adapt = true,
               double delta = 0.651,
               double gamma = 0.05,
               BaseRNG rand_int = BaseRNG(std::time(0)),
               const std::vector<double>* params_r = 0,
               const std::vector<int>* params_i = 0) 
        : adaptive_sampler(epsilon_adapt),
          _model(model),
          _epsilon(epsilon),
          _epsilon_pm(epsilon_pm),
          _epsilon_last(epsilon),
          _epsilon_adapt(epsilon_adapt),
          _delta(delta),
          _gamma(gamma),
          _da(gamma, std::vector<double>(1, 0.0)),
          _rand_int(rand_int),
          _rand_unit_norm(_rand_int, boost::normal_distribution<>()),
          _rand_uniform_01(_rand_int),
          _x(model.num_params_r()),
          _z(model.num_params_i()),
          _g(model.num_params_r())
      {
        model.init(_x,_z);
        if (params_r) {
          assert(params_r->size() == model.num_params_r());
          _x = *params_r;
        }
        if (params_i) {
          assert(params_i->size() == model.num_params_i());
          _z = *params_i;
        }
        _logp = model.grad_log_prob(_x,_z,_g);
        if (epsilon_adapt)
          find_reasonable_parameters();
      }

      virtual ~hmc_base() { }

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
        assert(x.size() == this->_x.size());
        assert(z.size() == this->_z.size());
        this->_x = x;
        this->_z = z;
        this->_logp = this->_model.grad_log_prob(this->_x,this->_z,this->_g);
      }

      /**
       * Sets the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parrs
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @throw std::invalid_argument if the number of real parameters does
       *   not match the number of parameters defined by the model.
       */
      void set_params_r(const std::vector<double>& x) {
        if (x.size() != this->_model.num_params_r())
          throw std::invalid_argument("x.size() must match num model params.");
        this->_x = x;
        this->_logp = this->_model.grad_log_prob(this->_x,this->_z,this->_g);
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
        if (z.size() != this->_model.num_params_i())
          throw std::invalid_argument("z.size() must match num params");
        this->_z = z;
        this->_logp = this->_model.grad_log_prob(this->_x,this->_z,this->_g);
      }

      bool varying_epsilon() {
        return this->_epsilon_pm != 0;
      }

      /**
       * Turn off parameter adaptation. 
       *
       * Because we're using primal-dual averaging, once we're done
       * adapting we want to set epsilon=the _average_ value of
       * epsilon over each adaptation step. This results in a
       * lower-variance estimate of the optimal epsilon.
       */
      virtual void adapt_off() {
        if (!this->adapting()) return;
        adaptive_sampler::adapt_off();
        std::vector<double> result;
        this->_da.xbar(result);
        this->_epsilon = exp(result[0]);
      }

      /**
       * Write the step size into position 1 of the specified vector.
       *
       * @param[out] params Where to store epsilon.
       */
      virtual void get_parameters(std::vector<double>& params) {
        params.assign(1, this->_epsilon);
      }






    }; // class hmc_base

  }
}
          

#endif
