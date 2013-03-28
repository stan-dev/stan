#ifndef __STAN__MCMC__STATIC__HMC__BETA__
#define __STAN__MCMC__STATIC__HMC__BETA__

#include <ctime>

#include <boost/random/mersenne_twister.hpp>

#include <stan/model/prob_grad_ad.hpp>

#include <stan/mcmc/adapter.hpp>
#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/hamiltonian.hpp>
#include <stan/mcmc/integrator.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    //////////////////////////////////////////////////
    //  HMC samplers with a static integration time //
    //////////////////////////////////////////////////
    
    // Unit metric
    
    template <typename M, class BaseRNG>
    class unit_metric_hmc: public hmc_base<M, 
                                           unit_e_point,
                                           unit_metric, 
                                           expl_leapfrog, 
                                           BaseRNG> {
      
    public:
      
      unit_metric_hmc(M &m, BaseRNG& rng);
      ~unit_metric_hmc() {};
      
      sample transition(sample& init_sample);
                                             
      void write_sampler_param_names(std::ostream& o);
      void write_sampler_params(std::ostream& o);
      void get_sampler_param_names(std::vector<std::string>& names);
      void get_sampler_params(std::vector<double>& values);
      
      void set_stepsize_and_T(const double e, const double t) {
        _epsilon = e; _T = t; _update_L();
      }
      
      void set_stepsize_and_L(const double e, const int l) {
        _epsilon = e; _L = l; _T = _epsilon * _L;
      }
      
      void set_T(const double t) { _T = t; _update_L(); }
      void set_stepsize(const double e) { _epsilon = e; _update_L(); }
      
      double get_stepsize() { return this->_epsilon; }
      double get_T() { return this->_T; }
      int get_L() { return this->_L; }
      
    protected:
      
      double _epsilon;
      double _T;
      int _L;
      
      void _update_L() { _L = static_cast<int>(_T / _epsilon); }
                        
    };
    
    template <typename M, class BaseRNG>
    unit_metric_hmc<M, BaseRNG>::unit_metric_hmc(M& m, BaseRNG& rng):
    hmc_base<M, unit_e_point, unit_metric, expl_leapfrog, BaseRNG>(m, rng),
    _epsilon(0.1),
    _T(1)
    { _update_L(); }
    
    template <typename M, class BaseRNG>
    sample unit_metric_hmc<M, BaseRNG>::transition(sample& init_sample) {
      
      unit_e_point z(init_sample.size_cont(), init_sample.size_disc());
      z.q = init_sample.cont_params();
      z.r = init_sample.disc_params();
      
      Eigen::VectorXd u(init_sample.size_cont());
      for (size_t i = 0; i < u.size(); ++i) u(i) = this->_rand_unit_gaus();
      
      this->_hamiltonian.sampleP(z, u);
      this->_hamiltonian.init(z);
      
      double H0 = this->_hamiltonian.H(z);
      
      for (int i = 0; i < _L; ++i) {
        this->_integrator.evolve(z, this->_hamiltonian, _epsilon);
      }
      
      double acceptProb = exp(H0 - this->_hamiltonian.H(z));
      
      double accept = true;
      if (acceptProb < 1 && this->_rand_uniform() > acceptProb) {
        z.q = init_sample.cont_params();
        accept = false;
      }
      
      return sample(z.q, z.r, - this->_hamiltonian.V(z), acceptProb);
      
    }
    
    template <typename M, class BaseRNG>
    void unit_metric_hmc<M, BaseRNG>::write_sampler_param_names(std::ostream& o) {
        o << "stepsize__,int_time__,";
    }
    
    template <typename M, class BaseRNG>
    void unit_metric_hmc<M, BaseRNG>::write_sampler_params(std::ostream& o) {
        o << this->_epsilon << "," << this->_T << ",";
    }
    
    template <typename M, class BaseRNG>
    void unit_metric_hmc<M, BaseRNG>::get_sampler_param_names(std::vector<std::string>& names) {
      names.clear();
      names.push_back("stepsize__");
      names.push_back("int_time__");
    }
    
    template <typename M, class BaseRNG>
    void unit_metric_hmc<M, BaseRNG>::get_sampler_params(std::vector<double>& values) {
      values.clear();
      values.push_back(this->_epsilon);
      values.push_back(this->_T);
    }
    
    template <typename M, class BaseRNG>
    class adapt_unit_metric_hmc: public unit_metric_hmc<M, BaseRNG>, public stepsize_adapter {
      
    public:
      
      adapt_unit_metric_hmc(M &m, BaseRNG& rng);
      ~adapt_unit_metric_hmc() {};
      
      sample transition(sample& init_sample);
      
    };

    template <typename M, class BaseRNG>
    adapt_unit_metric_hmc<M, BaseRNG>::adapt_unit_metric_hmc(M& m, BaseRNG& rng):
    unit_metric_hmc<M, BaseRNG>(m, rng),
    stepsize_adapter()
    {};
    
    template <typename M, class BaseRNG>
    sample adapt_unit_metric_hmc<M, BaseRNG>::transition(sample& init_sample) {
      
      sample s = unit_metric_hmc<M, BaseRNG>::transition(init_sample);
      
      if (this->_adapt_flag) {
        this->_learn_stepsize(this->_epsilon, s.accept_stat());
        this->_update_L();
      }
      
      return s;
      
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
      
      ps_point z(model.dim());
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
