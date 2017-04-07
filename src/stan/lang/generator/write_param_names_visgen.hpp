#ifndef STAN_LANG_GENERATOR_WRITE_PARAM_NAMES_VISGEN_HPP
#define STAN_LANG_GENERATOR_WRITE_PARAM_NAMES_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    // FIXME(carpenter): replace with simple templated function

    /**
     * Visitor to write parameter names into string vector
     * <code>names__</code>.
     */
    struct write_param_names_visgen : public visgen {
      /**
       * Construct a parameter names writer for the specified stream.
       *
       * @param[in,out] o stream for generating
       */
      explicit write_param_names_visgen(std::ostream& o)
        : visgen(o) {
      }

      /**
       * Generate the code to add the specified variable name to the
       * accumulator.
       *
       * @param[in] name variable name
       */
      void generate_param_names(const std::string& name) const {
        o_ << INDENT2 << "names__.push_back(\"" << name << "\");" << EOL;
      }

      void operator()(const nil& /*x*/) const  { }

      void operator()(const int_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const double_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const vector_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const simplex_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const ordered_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }
    };

  }
}
#endif
