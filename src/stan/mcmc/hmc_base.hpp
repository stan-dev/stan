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
       * @param adapt_epsilon Flag indicating whether adaptation is turned on.
       * @param rand_int Base random integer generator.
       */
      hmc_base(stan::model::prob_grad& model,
               double epsilon=-1,
               double epsilon_pm = 0.0,
               bool epsilon_adapt = true,
               double delta = 0.651,
               double gamma = 0.05,
               BaseRNG rand_int = BaseRNG(std::time(0))) 
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
        _logp = model.grad_log_prob(_x,_z,_g);
      }
    };

  }
}
          

#endif
