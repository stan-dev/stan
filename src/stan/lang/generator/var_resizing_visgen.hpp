#ifndef STAN_LANG_GENERATOR_VAR_RESIZING_VISGEN_HPP
#define STAN_LANG_GENERATOR_VAR_RESIZING_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_initialization.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Visitor to generate code to resize variables.
     */
    struct var_resizing_visgen : public visgen {
      /**
       * Indentation level.
       */
      size_t indent_;

      /**
       * Construct a variable resizing visitor generating to the
       * specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      explicit var_resizing_visgen(size_t indent, std::ostream& o)
        : visgen(o), indent_(indent) {  }

      void operator()(const nil& /*x*/) const { }  // dummy

      void operator()(const int_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "int", x.dims_,
                                nil(), nil());
      }

      void operator()(const double_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "double", x.dims_,
                                nil(), nil());
      }

      void operator()(const vector_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "vector_d", x.dims_,
                                x.M_, nil());
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "row_vector_d", x.dims_,
                                x.N_, nil());
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "vector_d", x.dims_,
                                x.K_, nil());
      }

      void operator()(const simplex_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "vector_d", x.dims_,
                                x.K_, nil());
      }

      void operator()(const ordered_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "vector_d", x.dims_,
                                x.K_, nil());
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "vector_d", x.dims_,
                                x.K_, nil());
      }

      void operator()(const matrix_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "matrix_d", x.dims_,
                                x.M_, x.N_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "matrix_d", x.dims_,
                                x.M_, x.N_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "matrix_d", x.dims_,
                                x.K_, x.K_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "matrix_d", x.dims_,
                                x.K_, x.K_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_initialization(o_, indent_, x.name_, "matrix_d", x.dims_,
                                x.K_, x.K_);
      }
    };

  }
}
#endif
