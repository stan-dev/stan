#ifndef STAN_LANG_GENERATOR_INIT_VARS_VISGEN_HPP
#define STAN_LANG_GENERATOR_INIT_VARS_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Variable initialization visitor.
     */
    struct init_vars_visgen : public visgen {
      /**
       * Construct a variable initialization visitor for generation
       * with the specified indentation level and stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      init_vars_visgen(int indent, std::ostream& o) : visgen(indent, o) { }

      /**
       * Generate code to fill the specified container with dummy
       * variable values.
       *
       * @tparam T type of variable declaration
       * @param[in] x variable declaration
       */
      template <typename T>
      void fill_real(const T& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_
           << ", std::numeric_limits<int>::min());"
           << EOL;
      }

      void operator()(const double_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const vector_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const row_vector_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const matrix_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const unit_vector_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const simplex_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const ordered_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        fill_real(x);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        fill_real(x);
      }
    };

  }
}
#endif
