#ifndef __STAN__MCMC__PS_POINT__BETA__
#define __STAN__MCMC__PS_POINT__BETA__

#include <fstream>

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace mcmc {

    // Point in a generic phase space
    class ps_point {
      
    public:
      
      ps_point(int n, int m): q(n), r(m), p(Eigen::VectorXd::Zero(n)),
                              V(0), g(Eigen::VectorXd::Zero(n))
      {};
      
      std::vector<double> q;
      std::vector<int> r;
      Eigen::VectorXd p;
      
      double V;
      Eigen::VectorXd g;
      
      void copy_base(ps_point& z) {
        q = z.q;
        r = z.r;
        p = z.p;
        V = z.V;
        g = z.g;
      }
      
      void write_header(std::ostream& o) {
        o << q.size() << " continuous, unconstrained parameters" << std::endl;
        o << r.size() << " discrete parameters" << std::endl;
        o << std::endl;
      }
        
      virtual void write_names(std::ostream& o) {
        o << "V";
        for(size_t i = 0; i < r.size(); ++i) o << ",disc_" << i;
        for(size_t i = 0; i < q.size(); ++i) o << ",cont_" << i;
        for(size_t i = 0; i < q.size(); ++i) o << ",p_cont_" << i;
        for(size_t i = 0; i < q.size(); ++i) o << ",g_cont_" << i;
        o << std::flush;
      }

      virtual void write(std::ostream& o) {
        o << V;
        for(size_t i = 0; i < r.size(); ++i) o << "," << r.at(i);
        for(size_t i = 0; i < q.size(); ++i) o << "," << q.at(i);
        for(size_t i = 0; i < q.size(); ++i) o << "," << p(i);
        for(size_t i = 0; i < q.size(); ++i) o << "," << g(i);
        o << std::flush;
      }
      
      virtual void write_metric(std::ostream& o) {
        o << "# No free parameters for unit metric" << std::endl;
      };
      
    };

  } // mcmc
  
} // stan


#endif