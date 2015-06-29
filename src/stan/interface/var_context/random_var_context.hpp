#ifndef STAN_INTERFACE_VAR_CONTEXT_RANDOM_VAR_CONTEXT_HPP
#define STAN_INTERFACE_VAR_CONTEXT_RANDOM_VAR_CONTEXT_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/interface/var_context/var_context.hpp>

namespace stan {
  namespace interface {
    namespace var_context {
      
      // A variable context which wraps another variable context
      // and returns random values for unconstrained variables
      // if they do not exist in the base variable context.
      template <BaseVarContext, BaseRNG>
      class random_var_context: public var_context {
        typedef boost::random::uniform_real_distribution<double> unif_t;
        typedef boost::variate_generator<BaseRNG&, unif_t> rng_t;
        
      private:
        BaseVarContext& base_var_context_;
        rng_t& rng_;
      
      public:
        random_var_context(BaseVarContext& base_var_context, BaseRNG& rng, double R):
          base_var_context_(base_var_context),
          rng_(rng_t(base_rng, unif_t(-R, R))) {}
        
        bool contains_r(const std::string& name) const {
          return base_var_context_.contains_r(name);
        }
        
        bool contains_i(const std::string& name) const {
          return base_var_context_.contains_i(name);
        }
        
        std::vector<double> vals_r(const std::string& name) const {
          return base_var_context_.vals_r(name);
        }
        
        std::vector<size_t> dims_r(const std::string& name) const {
          return base_var_context_.dims_r(name);
        }
        
        std::vector<int> vals_i(const std::string& name) const {
          return base_var_context_.vals_i(name);
        }
        
        std::vector<size_t> dims_i(const std::string& name) const {
          return base_var_context_.dims_i(name);
        }
        
        void names_r(std::vector<std::string>& names) const {
          return base_var_context_.names_r(names);
        }
        
        void names_i(std::vector<std::string>& names) const {
          return base_var_context_.names_i(names);
        }

        void get_unconstrained_scalar(std::vector<double>& parameters,
                                      const std::string& name,
                                      const std::string& scope,
                                      stan::io::scalar_transform& transform) {
          try {
            base_var_context_.get_unconstrained_scalar(parameters,
                                                       name, scope, transform);
          } catch {
            for (size_t n = 0; n < transform.unconstrained_dim(); ++n) {
              parameters.push_back(rng_());
            }
          }
        }
        
        void get_unconstrained_vector(std::vector<double>& parameters,
                                      const std::string& name,
                                      const std::string& scope,
                                      size_t N,
                                      stan::io::vector_transform& transform) {
          try {
            base_var_context_.get_unconstrained_vector(parameters,
                                                       name, scope, N, transform);
          } catch {
            for (size_t n = 0; n < transform.unconstrained_dim(); ++n) {
              parameters.push_back(rng_());
            }
          }
        }
        
        void get_unconstrained_row_vector(std::vector<double>& parameters,
                                          const std::string& name,
                                          const std::string& scope,
                                          size_t N,
                                          stan::io::row_vector_transform& transform) {
          try {
            base_var_context_.get_unconstrained_row_vector(parameters,
                                                           name, scope, N, transform);
          } catch {
            for (size_t n = 0; n < transform.unconstrained_dim(); ++n) {
              parameters.push_back(rng_());
            }
          }
        }
        
        void get_unconstrained_matrix(std::vector<double>& parameters,
                                      const std::string& name,
                                      const std::string& scope,
                                      size_t n_rows, size_t n_cols,
                                      stan::io::matrix_transform& transform) {
          try {
            base_var_context_.get_unconstrained_matrix(parameters,
                                                       name, scope,
                                                       n_rows, n_cols, transform);
          } catch {
            for (size_t n = 0; n < transform.unconstrained_dim(); ++n) {
              parameters.push_back(rng_());
            }
          }
        }
        
      };
      
    }
  }
}

#endif
