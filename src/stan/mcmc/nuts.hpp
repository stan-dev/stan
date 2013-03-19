#ifndef __STAN__MCMC__NUTS_H__
#define __STAN__MCMC__NUTS_H__

#include <ctime>
#include <cstddef>
#include <iostream>
#include <vector>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/math/util.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/dualaverage.hpp>
#include <stan/mcmc/hamiltonian.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    // EMHMC NUTS
    
    class flatNUTS: public baseNUTS<denseConstMetric>
    {
      
    public:
      
      flatNUTS(double epsilon, model &m);
      
    private:
      
      bool _compute_criterion(ps_point& minus, ps_point& plus, VectorXd& rho);
      
    }
    
    flatNUTS::flatNUTS(double epsilon, model& m):
    baseNUTS(epsilon, m)
    {}
    
    flatNUTS::_compute_criterion(psPoint& minus, psPoint& plus, VectorXd& rho)
    {
      return rho.transpose() * _hamiltonian.Minv() * plus.p) > 0
      && rho.transpose() * _hamiltonian.Minv() * minus.p) > 0;
      
    }

    // RMHMC NUTS
    
    class softAbsNUTS: public baseNUTS<softAbsMetric>
    {
      
    public:
      
      softAbsNUTS(double epsilon, model &m);
      
    private:
      
      bool _compute_criterion(psPoint& minus, ps_point& plus, VectorXd& rho);
      
    }
    
    flatNUTS::flatNUTS(double epsilon, model& m):
    baseNUTS(epsilon, m)
    {}
    
    flatNUTS::_compute_criterion(psPoint& minus, psPoint& plus, VectorXd& rho)
    {
      return rho.dot( plus.lambda_dot_p() ) > 0
      && rho.dot( minus.lambda_dot_p() ) > 0;
    }


  }

}

#endif
