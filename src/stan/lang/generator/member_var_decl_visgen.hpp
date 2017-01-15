#ifndef STAN_LANG_GENERATOR_MEMBER_VAR_DECL_VISGEN_HPP
#define STAN_LANG_GENERATOR_MEMBER_VAR_DECL_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Visitor for generating code to declare member variables.
     */
    struct member_var_decl_visgen : public visgen {
      /**
       * Construct a member variable declaration visitor at the
       * specified indentation level writing to the specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      member_var_decl_visgen(int indent, std::ostream& o)
        : visgen(indent, o) { }

      /**
       * Generate the array declaration for the specified generated
       * type, variable name, and number of array dimensions.
       *
       * @param[in] type string generated for variable type
       * @param[in] name name of variable
       * @param[in] size number of array dimensions for variable
       */
      void declare_array(const std::string& type, const std::string& name,
                         size_t size) const {
        for (int i = 0; i < indent_; ++i)
          o_ << INDENT;
        for (size_t i = 0; i < size; ++i)
          o_ << "vector<";
        o_ << type;
        if (size > 0)
          o_ << ">";
        for (size_t i = 1; i < size; ++i)
          o_ << " >";
        o_ << " " << name << ";" << EOL;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        declare_array("int", x.name_, x.dims_.size());
      }

      void operator()(const double_var_decl& x) const {
        declare_array("double", x.name_, x.dims_.size());
      }

      void operator()(const unit_vector_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const simplex_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const ordered_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const positive_ordered_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }

      void operator()(const cov_matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }

      void operator()(const corr_matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }

      void operator()(const vector_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const row_vector_var_decl& x) const {
        declare_array(("row_vector_d"), x.name_, x.dims_.size());
      }

      void operator()(const matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
    };

  }
}
#endif
