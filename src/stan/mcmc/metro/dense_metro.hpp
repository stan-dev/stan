#ifndef __STAN__MCMC__DENSE__METRO__HPP
#define __STAN__MCMC__DENSE__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

namespace stan {

  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class dense_metro: public base_metro<M, BaseRNG> {
      
    public:
      
      dense_metro(M& m, 
                  BaseRNG& rng, 
                  std::ofstream* error_msg)
        : base_metro<M, BaseRNG>(m, rng, error_msg),
          _prop_cov(Eigen::MatrixXd::Identity(m.num_params_r(), m.num_params_r())) { 
        this->_name = "Metropolis with a dense Euclidean metric"; 
        this->_nom_epsilon = 1;
      }

      void propose(std::vector<double>& q,
                   BaseRNG& rng) {
        Eigen::VectorXd zer(q.size());
        zer.setZero();

        Eigen::VectorXd prop(q.size());
        prop = this->_nom_epsilon 
          * stan::prob::multi_normal_rng(zer, _prop_cov, this->_rand_int);

        for(size_t i = 0; i < q.size(); i++)
          q[i] = prop(i);
      }                                  
                   
      void write_metric(std::ostream& o) {
        //o << "# Inverse covariance matrix elements:" << std::endl;
        o << "# Elements of inverse covariance matrix:" << std::endl;
        for(size_t i = 0; i < _prop_cov.rows(); ++i) {
          o << "# " << _prop_cov(i, 0) << std::flush;
          for(size_t j = 1; j < _prop_cov.cols(); ++j)
            o << ", " << _prop_cov(i, j) << std::flush;
          o << std::endl;
        }     
      };

    protected:

      Eigen::MatrixXd _prop_cov;
    };

  } // mcmc

} // stan
          

#endif
