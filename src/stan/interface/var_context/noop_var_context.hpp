#ifndef STAN_INTERFACE_VAR_CONTEXT_NOOP_VAR_CONTEXT_HPP
#define STAN_INTERFACE_VAR_CONTEXT_NOOP_VAR_CONTEXT_HPP

#include <stan/interface/var_context/var_context.hpp>

namespace stan {
  namespace interface {
    namespace var_context {

      class noop_var_context : public var_context {
      private:
        std::vector<double> const empty_vec_r_;
        std::vector<int> const empty_vec_i_;
        std::vector<size_t> const empty_vec_ui_;
        
      public:
        bool contains_r(const std::string& name) const {
          return false;
        }

        bool contains_i(const std::string& name) const {
          return false;
        }
        
        std::vector<double> vals_r(const std::string& name) const {
          return empty_vec_r_;
        }
        
        std::vector<size_t> dims_r(const std::string& name) const {
          return empty_vec_ui_;
        }
        
        std::vector<int> vals_i(const std::string& name) const {
          return empty_vec_i_;
        }
        
        std::vector<size_t> dims_i(const std::string& name) const {
          return empty_vec_ui_;
        }
        
        void names_r(std::vector<std::string>& names) const {
          names.resize(0);
        }
        
        void names_i(std::vector<std::string>& names) const {
          names.resize(0);
        }
        
      };
      

    }
  }
}

#endif
