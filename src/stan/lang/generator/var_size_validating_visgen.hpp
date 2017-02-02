#ifndef STAN_LANG_GENERATOR_VAR_SIZE_VALIDATING_VISGEN_HPP
#define STAN_LANG_GENERATOR_VAR_SIZE_VALIDATING_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Visitor for validating variable sizes.
     */
    struct var_size_validating_visgen : public visgen {
      /**
       * Processing stage.
       */
      const std::string stage_;

      /**
       * Construct a variable size validating visitor that generates
       * to the specified stream for the specified processing stage.
       *
       * @param[in,out] o stream for generating
       * @param[in] stage processing stage
       */
      var_size_validating_visgen(std::ostream& o, const std::string& stage)
        : visgen(o), stage_(stage) { }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "int", x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "double", x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.M_);
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "row_vector_d",
                                       x.dims_, x.N_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }

      void operator()(const simplex_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }

      void operator()(const ordered_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }

      void operator()(const matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.M_, x.N_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.M_, x.N_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }
    };

  }
}
#endif
