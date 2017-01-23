#ifndef STAN_LANG_GENERATOR_INIT_VISGEN_HPP
#define STAN_LANG_GENERATOR_INIT_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/var_size_validating_visgen.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor for generating code to initialize data variables from
     * an underying <code>var_context</code> in variable
     * <code>context__</code>.
     */
    struct init_visgen : public visgen {
      /**
       * Visitor for validating variable sizes
       */
      var_size_validating_visgen var_size_validator_;

      /**
       * Construct a visitor to generate initializations to the
       * specialized stream.
       *
       * @param[in,out] o stream for generating
       */
      explicit init_visgen(std::ostream& o)
        : visgen(o), var_size_validator_(o, "initialization") {  }

      /**
       * Generate the suffix for the appropriate unconstraining
       * functiont to map from constrained to unconstrained space.
       *
       * @tparam D type of variable declaration
       * @param[in] fun_prefix function name to be suffixed
       * @param[in] x variable declaration
       * @return function name plus constraints as string
       */
      template <typename D>
      std::string function_args(const std::string& fun_prefix, const D& x)
        const {
        std::stringstream ss;
        ss << fun_prefix;
        if (has_lub(x)) {
          ss << "_lub_unconstrain(";
          generate_expression(x.range_.low_.expr_, ss);
          ss << ',';
          generate_expression(x.range_.high_.expr_, ss);
          ss << ',';
        } else if (has_lb(x)) {
          ss << "_lb_unconstrain(";
          generate_expression(x.range_.low_.expr_, ss);
          ss << ',';
        } else if (has_ub(x)) {
          ss << "_ub_unconstrain(";
          generate_expression(x.range_.high_.expr_, ss);
          ss << ',';
        } else {
          ss << "_unconstrain(";
        }
        return ss.str();
      }

      /**
       * Generate the loop over the specified dimension sizes at the
       * specified indent level.
       *
       * @param[in] dims dimension sizes
       * @param[in] indent indentation level
       */
      void generate_dims_loop_fwd(const std::vector<expression>& dims,
                                  int indent = 2U) const {
        size_t size = dims.size();
        for (size_t i = 0; i < size; ++i) {
          generate_indent(i + indent, o_);
          o_ << "for (int i" << i << "__ = 0U; i" << i << "__ < ";
          generate_expression(dims[i].expr_, o_);
          o_ << "; ++i" << i << "__)" << EOL;
        }
        generate_indent(2U + dims.size(), o_);
      }

      /**
       * Generate variable and indexes for the specified variable
       * name and number of dimensions.
       *
       * @param[in] name variable name
       * @param[in] num_dims number of dimensions
       */
      void generate_name_dims(const std::string& name,
                              size_t num_dims) const {
        o_ << name;
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "[i" << i << "__]";
      }

      /**
       * Generate try-catch block to write varaibles into variable
       * <code>writer__</code>, using the specified method in the
       * writer, variable name, and dimension sizes.
       *
       * @param[in] write_method_name name of writer method
       * @param[in] var_name name of variable
       * @param[in] dims dimension sizes
       */
      void generate_write_loop(const std::string& write_method_name,
                               const std::string& var_name,
                               const std::vector<expression>& dims) const {
        generate_dims_loop_fwd(dims);
        o_ << "try {"
           << EOL
           << INDENT3
           << "writer__." << write_method_name;
        generate_name_dims(var_name, dims.size());
        o_ << ");"
           << EOL
           << INDENT2
           << "} catch (const std::exception& e) { "
           << EOL
           << INDENT3
           << "throw std::runtime_error("
           << "std::string(\"Error transforming variable "
           << var_name << ": \") + e.what());"
           << EOL
           << INDENT2
           << "}"
           << EOL;
      }

      /**
       * Generate the variable declarations for a variable with
       * specified name, base type, dimension sizes, optional
       * dimension sizes for vectors, row vectors and matrices, and
       * optional definition.
       *
       * @param[in] name variable name
       * @param[in] base_type basic type of variable
       * @param[in] dims array dimension sizes
       * @param[in] type_arg1 optional vector or row vector size or
       * matrix rows
       * @param[in] type_arg2 optional matrix columns
       * @param[in] definition optional definition
       */
      void generate_declaration(const std::string& name,
                                const std::string& base_type,
                                const std::vector<expression>& dims,
                                const expression& type_arg1 = expression(),
                                const expression& type_arg2 = expression(),
                                const expression& definition = expression())
      const {
        o_ << INDENT2 << "// generate_declaration " << name << std::endl;
        o_ << INDENT2;
        generate_type(base_type, dims, dims.size(), o_);
        o_ << ' ' << name;
        generate_initializer(o_, base_type, dims, type_arg1, type_arg2);
      }

      /**
       * Generate indentation based on number of array dimensions and
       * vector/matrix dimensions.
       *
       * @param[in] base_indent additional indentation level
       * @param[in] dims dimension sizes
       * @param[in] dim1 vector or row vector size or matrix rows or
       * nil if not vector or matrix
       * @param[in] dim2 matrix columns or nil if not matrix
       */
      void generate_indent_num_dims(size_t base_indent,
                                    const std::vector<expression>& dims,
                                    const expression& dim1,
                                    const expression& dim2) const {
        generate_indent(dims.size() + base_indent, o_);
        if (!is_nil(dim1)) o_ << INDENT;
        if (!is_nil(dim2)) o_ << INDENT;
      }

      /**
       * Generate the loop to buffer the values for the variable with
       * the specified name, base type, dimension sizes, vector/matrix
       * dimension sizes, and indentation level.
       *
       * @param[in] base_type base type of variable as string
       * @param[in] name variable name
       * @param[in] dims dimension sizes
       * @param[in] dim1 optional vector/row vector size or matrix
       * number of rows
       * @param[in] dim2 optional matrix number of columns
       * @param[in] indent optional indentation level
       */
      void generate_buffer_loop(const std::string& base_type,
                                const std::string& name,
                                const std::vector<expression>& dims,
                                const expression& dim1 = expression(),
                                const expression& dim2 = expression(),
                                int indent = 2U) const {
        size_t size = dims.size();
        bool is_matrix = !is_nil(dim1) && !is_nil(dim2);
        bool is_vector = !is_nil(dim1) && is_nil(dim2);
        int extra_indent = is_matrix ? 2U : is_vector ? 1U : 0U;
        if (is_matrix) {
          generate_indent(indent, o_);
          o_ << "for (int j2__ = 0U; j2__ < ";
          generate_expression(dim2.expr_, o_);
          o_ << "; ++j2__)" << EOL;

          generate_indent(indent+1, o_);
          o_ << "for (int j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_, o_);
          o_ << "; ++j1__)" << EOL;
        } else if (is_vector) {
          generate_indent(indent, o_);
          o_ << "for (int j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_, o_);
          o_ << "; ++j1__)" << EOL;
        }
        for (size_t i = 0; i < size; ++i) {
          size_t idx = size - i - 1;
          generate_indent(i + indent + extra_indent, o_);
          o_ << "for (int i" << idx << "__ = 0U; i" << idx << "__ < ";
          generate_expression(dims[idx].expr_, o_);
          o_ << "; ++i" << idx << "__)" << EOL;
        }
        generate_indent_num_dims(2U, dims, dim1, dim2);
        o_ << name;
        for (size_t i = 0; i < dims.size(); ++i)
          o_ << "[i" << i << "__]";
        if (is_matrix)
          o_ << "(j1__,j2__)";
        else if (is_vector)
          o_ << "(j1__)";
        o_ << " = vals_" << base_type << "__[pos__++];" << EOL;
      }

      /**
       * Generate checks to make sure the specified integer is defined
       * in the variable context from which data is being read,
       * <code>context__</code>.
       *
       * @param[in] name name of integer variable
       */
      void generate_check_int(const std::string& name) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_i(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2 << "vals_i__ = context__.vals_i(\"" << name << "\");"
           << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }

      /**
       * Generate code to check that double variable of specified name
       * exists in the variable context <code>context__</code>.
       *
       * @param[in] name variable name
       */
      void generate_check_double(const std::string& name) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_r(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << name << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }

      void operator()(const double_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "double", x.dims_, nil(), nil(), x.def_);
        if (is_nil(x.def_)) {
          generate_buffer_loop("r", x.name_, x.dims_);
        }
        generate_write_loop(function_args("scalar", x),
                            x.name_, x.dims_);
      }
      void operator()(const nil& /*x*/) const { }  // dummy

      void operator()(const int_var_decl& x) const {
        generate_check_int(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "int", x.dims_, nil(), nil(), x.def_);
        generate_buffer_loop("i", x.name_, x.dims_);
        generate_write_loop("integer(", x.name_, x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.M_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_);
        generate_write_loop(function_args("vector", x),
                            x.name_, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "row_vector_d", x.dims_, x.N_, nil(),
                             x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.N_);
        generate_write_loop(function_args("row_vector", x),
                            x.name_, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop(function_args("matrix", x),
                            x.name_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("unit_vector_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("simplex_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("ordered_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("positive_ordered_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop("cholesky_factor_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cholesky_corr_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cov_matrix_unconstrain(", x.name_, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_check_double(x.name_);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("corr_matrix_unconstrain(", x.name_, x.dims_);
      }
    };

  }
}
#endif
