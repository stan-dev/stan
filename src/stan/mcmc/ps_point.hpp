#ifndef __STAN__MCMC__PS_POINT__BETA__
#define __STAN__MCMC__PS_POINT__BETA__

#include <vector>
#include <Eigen/Dense>

namespace stan {
  
  namespace mcmc {

    // Point in a generic phase space
    class ps_point {
      
    public:
      
      ps_point(int n, int m): q(n), r(m), p(Eigen::VectorXd::Zero(n)) 
        {};
      
      std::vector<double> q;
      std::vector<int> r;
      Eigen::VectorXd p;
      
    };

  } // mcmc
  
} // stan


#endif