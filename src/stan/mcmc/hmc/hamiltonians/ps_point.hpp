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
      
      ps_point(int n): q(n), p(n), V(0), g(n) {};
  
      ps_point(const ps_point& z): q(z.q.size()), p(z.p.size()), V(z.V), g(z.g.size())
      {
        _fast_vector_copy<double>(q, z.q);
        _fast_vector_copy<double>(p, z.p);
        _fast_vector_copy<double>(g, z.g);
      }
      
      
      ps_point& operator= (const ps_point& z)
      {
        
        if(this == &z) return *this;
        
        _fast_vector_copy<double>(q, z.q);
        
        V = z.V;
        
        _fast_vector_copy<double>(p, z.p);
        _fast_vector_copy<double>(g, z.g);
        
        return *this;
        
      }
      
      Eigen::VectorXd q;
      Eigen::VectorXd p;
      
      double V;
      Eigen::VectorXd g;
        
      virtual void get_param_names(std::vector<std::string>& model_names,
                                   std::vector<std::string>& names) {
        for(int i = 0; i < q.size(); ++i)
          names.push_back(model_names.at(i));
        for(int i = 0; i < q.size(); ++i)
          names.push_back(std::string("p_") + model_names.at(i));
        for(int i = 0; i < q.size(); ++i)
          names.push_back(std::string("g_") + model_names.at(i));
      }

      virtual void get_params(std::vector<double>& values) {
        for(int i = 0; i < q.size(); ++i)
          values.push_back(q(i));
        for(int i = 0; i < q.size(); ++i)
          values.push_back(p(i));
        for(int i = 0; i < q.size(); ++i)
          values.push_back(g(i));
      }
      
      virtual void write_metric(std::ostream* o) {
        if(!o) return;
        *o << "# No free parameters for unit metric" << std::endl;
      }
      
    protected:
      
      template <typename T>
      inline void _fast_vector_copy(Eigen::Matrix<T, Eigen::Dynamic, 1>& v_to, const Eigen::Matrix<T, Eigen::Dynamic, 1>& v_from) {
        v_to.resize(v_from.size());
        std::memcpy(&v_to(0), &v_from(0), v_from.size() * sizeof(double));
      }

      template <typename T>
      inline void _fast_matrix_copy(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& v_to,
                                    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& v_from) {
        v_to.resize(v_from.rows(), v_from.cols());
        std::memcpy(&v_to(0), &v_from(0), v_from.size() * sizeof(double));
      }
      
    };

  } // mcmc
  
} // stan


#endif