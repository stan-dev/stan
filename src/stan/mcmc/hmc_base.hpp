#ifndef __STAN__MCMC__HMC_BASE_H__
#define __STAN__MCMC__HMC_BASE_H__

#include <ctime>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/adaptive_sampler.hpp>

namespace stan {

  namespace mcmc {

    template <class BaseRNG = boost::mt19937>
    class hmc_base : public adaptive_sampler {
    protected:
      BaseRNG _rand_int;
      boost::variate_generator<BaseRNG&, boost::normal_distribution<> >
        _rand_unit_norm;
      boost::uniform_01<BaseRNG&> _rand_uniform_01;
    public:
      /**
       * Construct a base HMC sampler.
       *
       * @param adapt Flag indicating whether adaptation is turned on.
       * @param rand_int Base random integer generator.
       */
      hmc_base(bool adapt,
               BaseRNG rand_int = BaseRNG(std::time(0))) 
        : adaptive_sampler(adapt),
          _rand_int(rand_int),
          _rand_unit_norm(_rand_int, boost::normal_distribution<>()),
          _rand_uniform_01(_rand_int)
      {  }
    };

  }
}
          

#endif
