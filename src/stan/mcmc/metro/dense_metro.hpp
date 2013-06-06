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
                  std::ostream* o = &std::cout, 
                  std::ostream* e = 0)
        : base_metro<M, BaseRNG>(m, rng, o, e),
          _prop_cov(Eigen::MatrixXd::Identity(m.num_params_r(), 
                                              m.num_params_r())),
          _prop_cov_inv(Eigen::MatrixXd::Identity(m.num_params_r(), 
                                                  m.num_params_r())) { 
        this->_name = "Metropolis with a dense metric"; 
        this->_nom_epsilon = 1;
      }

      void propose(std::vector<double>& q,
                   BaseRNG& rng) {
        Eigen::VectorXd prop(q.size());
 
        for(size_t  i = 0; i < q.size(); i++)
          prop(i) = stan::prob::normal_rng(0,
                                           this->_nom_epsilon, 
                                           rng);

        prop = _prop_cov_inv.llt().matrixL() * prop;

        for(size_t i = 0; i < q.size(); i++)
          q[i] += prop(i);
      }                                  
                   
      void write_metric(std::ostream& o) {
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
      Eigen::MatrixXd _prop_cov_inv;
    };

  } // mcmc

} // stan
          

#endif
