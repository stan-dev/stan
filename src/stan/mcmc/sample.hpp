#ifndef __STAN__MCMC__SAMPLE__HPP__
#define __STAN__MCMC__SAMPLE__HPP__

#include <vector>

namespace stan {
  
  namespace mcmc {
    
    class sample {
      
    private:
      
      std::vector<double> _cont_params; // Continuous coordinates of sample
      std::vector<int> _disc_params;    // Discrete coordinates of sample
      double _log_prob;                 // Log probability of sample
      double _accept_stat;              // Acceptance statistic of transition
      
    public:
      
      sample(const std::vector<double>& q,
             const std::vector<int>& r,
             double log_prob,
             double stat) :
      _cont_params(q), 
      _disc_params(r),
      _log_prob(log_prob),
      _accept_stat(stat) {};
      
      virtual ~sample() {}; // No-op
      
      inline int size_cont() const { 
        return _cont_params.size(); 
      }
      
      inline double cont_params(int k) const { 
        return _cont_params[k]; 
      }
      
      inline void cont_params(std::vector<double>& x) const {
        x = _cont_params;
      }
      
      inline const std::vector<double>& cont_params() const { 
        return _cont_params; 
      }
      
      inline int size_disc() const { 
        return _disc_params.size();
      }
      
      inline int disc_params(int k) const {
        return _disc_params[k];
      }
      
      inline void disc_params(std::vector<int>& n) const {
        n = _disc_params;
      }
      
      inline const std::vector<int>& disc_params() const {
        return _disc_params;
      }
      
      inline double log_prob() const {
        return _log_prob;
      }
      
      inline double accept_stat() const {
        return _accept_stat;
      }
      
    };
    
  } // mcmc
  
} // stan

#endif

