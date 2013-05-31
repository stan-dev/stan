#ifndef __STAN__MCMC__PS_POINT__BETA__
#define __STAN__MCMC__PS_POINT__BETA__

#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace mcmc {

    // Point in a generic phase space
    class ps_point {
      
    public:
      
      ps_point(int n, int m): q(n), r(m), p(n), V(0), g(n) {
        p.setZero();
        g.setZero();
      };
  
      ps_point(const ps_point& z): q(z.q), r(z.r), p(z.p.size()), V(z.V), g(z.g.size())
      {
        std::memcpy(&p(0), &(z.p(0)), z.p.size() * sizeof(double));
        std::memcpy(&g(0), &(z.g(0)), z.g.size() * sizeof(double));
      }
      
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
        
      virtual void get_param_names(std::vector<std::string>& model_names,
                                   std::vector<std::string>& names) {
        for(size_t i = 0; i < q.size(); ++i)
          names.push_back(model_names.at(i));
        for(size_t i = 0; i < q.size(); ++i)
          names.push_back(std::string("p_") + model_names.at(i));
        for(size_t i = 0; i < q.size(); ++i)
          names.push_back(std::string("g_") + model_names.at(i));
      }

      virtual void get_params(std::vector<double>& values) {
        for(size_t i = 0; i < q.size(); ++i)
          values.push_back(q.at(i));
        for(size_t i = 0; i < q.size(); ++i)
          values.push_back(p(i));
        for(size_t i = 0; i < q.size(); ++i)
          values.push_back(g(i));
      }
      
      virtual void write_metric(std::ostream* o) {
        if(!o) return;
        *o << "# No free parameters for unit metric" << std::endl;
      };
      
    };

  } // mcmc
  
} // stan


#endif