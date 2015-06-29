#ifndef STAN_INTERFACE_VAR_CONTEXT_EIGEN_VAR_CONTEXT_HPP
#define STAN_INTERFACE_VAR_CONTEXT_EIGEN_VAR_CONTEXT_HPP

#include <stan/interface/var_context/typedefs.hpp>
#include <stan/interface/var_context/map_var_context.hpp>

namespace stan {
  namespace interface {
    namespace var_context {

      // Extends a map_var_context to accept Eigen objects
      class eigen_var_context : public map_var_context {
      public:
        
        void add_vector(const std::string& name,
                        const vector_t& v,
                        std::ostream *o) {
          
          std::vector<double> values;
          for (int n = 0; n < v.size(); ++n)
            values.push_back(v(n));
          
          std::vector<size_t> dims;
          dims.push_back(v.size());
          
          add_r(name, values, dims, o);
        }
        
        void add_row_vector(const std::string& name,
                            const row_vector_t& v,
                            std::ostream *o) {
          
          std::vector<double> values;
          for (int n = 0; n < v.size(); ++n)
            values.push_back(v(n));
          
          std::vector<size_t> dims;
          dims.push_back(v.size());
          
          add_r(name, values, dims, o);
        }
        
        void add_matrix(const std::string& name,
                        const matrix_t& m,
                        std::ostream *o) {
          
          std::vector<double> values;
          for (int j = 0; j < m.cols(); ++j)
            for (int i = 0; i < m.rows(); ++i)
              values.push_back(m(i, j));
          
          std::vector<size_t> dims;
          dims.push_back(m.rows());
          dims.push_back(m.cols());
          
          add_r(name, values, dims, o);
        }
        
      };
      

    }
  }
}

#endif
