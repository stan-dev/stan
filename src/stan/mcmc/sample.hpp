#ifndef __STAN__MCMC__SAMPLE__HPP__
#define __STAN__MCMC__SAMPLE__HPP__

#include <vector>
#include <string>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace mcmc {
    
    class sample {
      
    private:
      
      Eigen::VectorXd _cont_params; // Continuous coordinates of sample
      double _log_prob;             // Log probability of sample
      double _accept_stat;          // Acceptance statistic of transition
      
    public:
      
      sample(const Eigen::VectorXd& q,
             double log_prob,
             double stat) :
      _cont_params(q),
      _log_prob(log_prob),
      _accept_stat(stat) {};
      
      virtual ~sample() {}; // No-op
      
      int size_cont() const {
        return _cont_params.size(); 
      }
      
      double cont_params(int k) const {
        return _cont_params(k);
      }
      
      void cont_params(Eigen::VectorXd& x) const {
        x = _cont_params;
      }
      
      const Eigen::VectorXd& cont_params() const {
        return _cont_params; 
      }
      
      inline double log_prob() const {
        return _log_prob;
      }
      
      inline double accept_stat() const {
        return _accept_stat;
      }
      
      void get_sample_param_names(std::vector<std::string>& names) {
        names.push_back("lp__");
        names.push_back("accept_stat__");
      }
      
      void get_sample_params(std::vector<double>& values) {
        values.push_back(_log_prob);
        values.push_back(_accept_stat);
      }
      
    };
    
  } // mcmc
  
} // stan

#endif

