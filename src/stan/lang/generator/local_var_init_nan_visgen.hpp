#ifndef STAN_LANG_GENERATOR_LOCAL_VAR_INIT_NAN_VISGEN_HPP
#define STAN_LANG_GENERATOR_LOCAL_VAR_INIT_NAN_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/visgen.hpp>

namespace stan {
  namespace lang {


    /**
     * Visitor to generate local variable initializations.
     */
    struct local_var_init_nan_visgen : public visgen {
      /**
       * true if the generation takes place in a variable context.
       */
      const bool is_var_context_;

      /**
       * Construct a local variable initializer in the specified
       * context at the specified indentation level and writing to the
       * specified stream.
       *
       * @param[in] is_var_context true if generation is in variable
       * context
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      local_var_init_nan_visgen(bool is_var_context, int indent,
                                std::ostream& o)
        : visgen(indent, o), is_var_context_(is_var_context) { }

      /**
       * Initialize the variable or its contained elements to
       * not-a-number or the dummy variable if it's in a variable
       * context.
       *
       * @tparam T type of variable declaration
       * @param x variable declaration
       */
      template <typename T>
      void generate_init(const T& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::initialize(" << x.name_ << ", "
           << (is_var_context_
               ? "DUMMY_VAR__"
               : "std::numeric_limits<double>::quiet_NaN()")
           << ");" << EOL;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& /*x*/) const { }

      void operator()(const double_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const vector_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const matrix_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const simplex_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const ordered_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_init(x);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_init(x);
      }
    };

  }
}
#endif
