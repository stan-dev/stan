#ifndef __STAN__MCMC__STATIC__HMC__BETA__
#define __STAN__MCMC__STATIC__HMC__BETA__

#include <ctime>

#include <boost/random/mersenne_twister.hpp>

#include <stan/model/prob_grad.hpp>

#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    //////////////////////////////////////////////////
    //  HMC samplers with a static integration time //
    //////////////////////////////////////////////////
    
    // Unit metric
    
    template <class BaseRNG = boost::mt19937>
    class unit_metric_hmc: public hmc_base<prob_grad, unit_metric, expl_leapfrog, BaseRNG> {
      
    public:
      
      unit_metric(prob_grad &m);
      ~unit_metric() {};
      
      void sample(VectorXd& q);
      
      void set_stepsize_and_T(const double e, const double t) {
        _epsilon = e; _T = t; _update_L();
      }
      
      void set_stepsize_and_L(const double e, const double l) {
        _epsilon = e; _L = l; _T = _epsilon * _L;
      }
      
      void set_T(const double t) { _T = t; _update_L(); }
      void set_stepsize(const double e) { _epsilon = e; _update_L(); }
      
      void get_stepsize() { return _epsilon; }
      void get_L() { return _L; }
      void get_T() { return _T; }
      
    private:
      
      double _epsilon;
      double _L;
      double _T;
      
      _update_L() { _L = static_cast<int>(_T / _epsilon); }
                        
    };
    
    unit_metric_hmc::unit_metric_hmc(model& m):
    hmc_base(m),
    _epsilon(0.1),
    _T(1)
    { _update_L(); }
    
    void basicHMC::sample(VectorXd& q) {
      
      psPoint z(q.size());
      z.q = q;
      
      VectorXd u(q.size());
      for (int i = 0; i < u.size(); ++i) u(i) = _rand_unit_gaus();
      
      _hamiltonian.sampleP(z.p, u);
      
      double H0 = _hamiltonian.H(z);
      
      for(int i = 0; i < _L; ++i)
      {
        _integrator.evolve(z, _hamiltonian, _epsilon);
      }
      
      double acceptProb = exp(H0 - _hamiltonian.H(z));
      
      if(_rand_uniform() < acceptProb) q = z.q();
      
      return;
      
    }
    
    /*
    //////////////////////////////////////////////////
    //     Basic Leapfrog with Custom Mass Matrix   //
    //////////////////////////////////////////////////
    
    class EMHMC: public hmcSampler<denseConstMetric>
    {
      
    public:
      
      EMHMC(double epsilon, double L, model &m);
      
      void sample(VectorXd& q);
      
    private:
      
      double _epsilon;
      double _L;
      
    }
    
    EMHMC::EMHMC(double epsilon, double L, model& m):
    hmcSampler(m),
    _epsilon(epsilon),
    _L(L)
    {}
    
    void EMHMC::sample(VectorXd& q)
    {
      
      _hamiltonian.Minv = MatrixXd::Ones(model.dim(), model.dim());
      
      psPoint z(model.dim());
      z.q = q;
      _hamiltonian.sampleP(z.p, RNG);
      
      double H0 = _hamiltonian.H(z);
      
      for(int i = 0; i < _L; ++i)
      {
        _evolve.evolve(z, _hamiltonian, _epsilon);
      }
      
      double acceptProb = exp(H0 - _hamiltonian.H(z));
      
      if(RNG.U() < acceptProb) q = z.q();
      
      return;
      
    }
     */

    
  } // mcmc

} // stan
          

#endif
