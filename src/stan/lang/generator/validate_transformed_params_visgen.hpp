#ifndef STAN_LANG_GENERATOR_VALIDATE_TRANSFORMED_PARAMS_VISGEN_HPP
#define STAN_LANG_GENERATOR_VALIDATE_TRANSFORMED_PARAMS_VISGEN_HPP

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
     * Visitor for generating code to validate transformed
     * parameters.
     */
    struct validate_transformed_params_visgen : public visgen {
      /**
       * Construct a visitor for validating transformed parameters
       * with the specified indentation level and writing to the
       * specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      validate_transformed_params_visgen(int indent, std::ostream& o)
        : visgen(indent, o) { }

      /**
       * Generate code to validate the array component of the variable
       * with the specified name, array dimensions, and number of
       * matrix dimensions.
       *
       * @param[in] name variable name
       * @param[in] dims size of dimensions
       * @param matrix_dims number of matrix dimensions
       */
      void validate_array(const std::string& name,
                          const std::vector<expression>& dims,
                          size_t matrix_dims) const {
        for (size_t k = 0; k < dims.size(); ++k) {
          generate_indent(indent_ + k, o_);
          o_ << "for (int i" << k << "__ = 0; i" << k << "__ < ";
          generate_expression(dims[k], o_);
          o_ << "; ++i" << k << "__) {" << EOL;
        }

        size_t non_matrix_dims = dims.size() - matrix_dims;
        generate_indent(indent_ + dims.size(), o_);
        o_ << "if (stan::math::is_uninitialized(" << name;
        for (size_t k = 0; k < non_matrix_dims; ++k)
          o_ << "[i" << k << "__]";
        if (matrix_dims > 0) {
          o_ << "(i" << non_matrix_dims << "__";
          if (matrix_dims > 1)
            o_ << ",i" << (non_matrix_dims + 1) << "__";
          o_ << ')';
        }
        o_ << ")) {" << EOL;
        generate_indent(indent_ + dims.size() + 1, o_);
        o_ << "std::stringstream msg__;" << EOL;
        generate_indent(indent_ + dims.size() + 1, o_);
        o_ << "msg__ << \"Undefined transformed parameter: "
           << name << "\"";
        for (size_t k = 0; k < dims.size(); ++k) {
          o_ << " << '['";
          o_ << " << i" << k << "__";
          o_ << " << ']'";
        }
        o_ << ';' << EOL;
        generate_indent(indent_ + dims.size() + 1, o_);
        o_ << "throw std::runtime_error(msg__.str());" << EOL;

        generate_indent(indent_ + dims.size(), o_);
        o_ << "}" << EOL;
        for (size_t k = 0; k < dims.size(); ++k) {
          generate_indent(indent_ + dims.size() - k - 1, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        validate_array(x.name_, dims, 0);
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        validate_array(x.name_, dims, 0);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 1);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 2);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 2);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }
    };

  }
}
#endif
