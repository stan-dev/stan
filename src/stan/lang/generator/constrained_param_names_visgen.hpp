#ifndef STAN_LANG_GENERATOR_CONSTRAINED_PARAM_NAMES_VISGEN_HPP
#define STAN_LANG_GENERATOR_CONSTRAINED_PARAM_NAMES_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor to push constrained parameter names onto
     * <code>param_names__</code>
     */
    struct constrained_param_names_visgen : public visgen {
      /**
       * Construct a constrained parameter names generator for the
       * specified stream.
       */
      explicit constrained_param_names_visgen(std::ostream& o) : visgen(o) { }

      /**
       * Generate the parameter names for the specified parameter
       * name, with specified matrix and array dimension sizes.
       *
       * @param[in] matrix_dims sequence of matrix dimension sizes
       * @param[in] name variable name
       * @param[in] dims sequence of array dimension sizes
       */
      void
      generate_param_names_array(const std::vector<expression>& matrix_dims,
                                 const std::string& name,
                                 const std::vector<expression>& dims) const {
        // begin for loop dims
        std::vector<expression> combo_dims(dims);
        for (size_t i = 0; i < matrix_dims.size(); ++i)
          combo_dims.push_back(matrix_dims[i]);

        for (size_t i = combo_dims.size(); i-- > 0; ) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "for (int k_" << i << "__ = 1;"
             << " k_" << i << "__ <= ";
          generate_expression(combo_dims[i].expr_, o_);
          o_ << "; ++k_" << i << "__) {" << EOL;  // begin (1)
        }

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__.str(std::string());" << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__ << \"" << name << '"';

        for (size_t i = 0; i < combo_dims.size(); ++i)
          o_ << " << '.' << k_" << i << "__";
        o_ << ';' << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_names__.push_back(param_name_stream__.str());" << EOL;

        // end for loop dims
        for (size_t i = 0; i < combo_dims.size(); ++i) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "}" << EOL;  // end (1)
        }
      }

      void operator()(const nil& /*x*/) const  { }

      void operator()(const int_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
    };

  }
}
#endif
