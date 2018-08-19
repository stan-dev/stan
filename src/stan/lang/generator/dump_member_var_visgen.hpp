#ifndef STAN_LANG_GENERATOR_DUMP_MEMBER_VAR_VISGEN_HPP
#define STAN_LANG_GENERATOR_DUMP_MEMBER_VAR_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/var_resizing_visgen.hpp>
#include <stan/lang/generator/var_size_validating_visgen.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    // FIXME(carpenter): factor repeated code into functions

    /**
     * Visitor for generating code to read member variables (data)
     * from the dump format through a variable context.
     */
    struct dump_member_var_visgen : public visgen {
      /**
       * Indentation level.
       */
      size_t indent_;

      /**
       * Visitor for generating code to resize variables.
       */
      var_resizing_visgen var_resizer_;

      /**
       * Visitor for generating code to validate variable sizes.
       */
      var_size_validating_visgen var_size_validator_;

      /**
       * Construct a visitor to generate code to read member variables
       * to the specified stream.
       *
       * @param[in] indent indentation level
       * @param[in,out] o stream for generating
       */
      explicit dump_member_var_visgen(size_t indent, std::ostream& o)
        : visgen(o), indent_(indent),
          var_resizer_(var_resizing_visgen(indent, o)),
          var_size_validator_(var_size_validating_visgen(indent, o,
                                                    "data initialization")) {
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_i__ = context__.vals_i(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_"
             << dim << "__ < " << x.name_ << "_limit_" << dim
             << "__; ++i_" << dim << "__) {" << EOL;
          ++indentation;
        }
        generate_indent(indentation, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_i__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(--indentation, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
          ++indentation;
        }
        generate_indent(indentation, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(--indentation, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.M_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.N_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = indent_;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = indent_ + 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "}" << EOL;
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_
           << "_k_mat_lim__; ++n_mat__) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_
           << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = indent_ + 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "}" << EOL;
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = indent_ + 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "}" << EOL;
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = indent_ + 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "}" << EOL;
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        generate_indent(indent_, o_);
        o_ << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");"
           << EOL;
        generate_indent(indent_, o_);
        o_ << "pos__ = 0;" << EOL;
        generate_indent(indent_, o_);
        o_ << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, NOT_USER_FACING, o_);
        o_ << ";" << EOL;
        generate_indent(indent_, o_);
        o_ << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
        generate_indent(indent_ + 1, o_);
        o_ << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = indent_ + 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], NOT_USER_FACING, o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "}" << EOL;
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }
    };

  }
}
#endif
