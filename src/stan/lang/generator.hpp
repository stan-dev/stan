#ifndef STAN_LANG_GENERATOR_HPP
#define STAN_LANG_GENERATOR_HPP


// FIXME(carpenter): move into AST
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>

// FIXME(carpenter): move into general utilities
#include <stan/lang/generator/to_string.hpp>

#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/printable_visgen.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>
#include <stan/lang/generator/init_local_var_visgen.hpp>
#include <stan/lang/generator/init_vars_visgen.hpp>
#include <stan/lang/generator/local_var_decl_visgen.hpp>
#include <stan/lang/generator/local_var_init_nan_visgen.hpp>
#include <stan/lang/generator/member_var_decl_visgen.hpp>
#include <stan/lang/generator/statement_visgen.hpp>
#include <stan/lang/generator/validate_var_decl_visgen.hpp>
#include <stan/lang/generator/validate_transformed_params_visgen.hpp>
#include <stan/lang/generator/var_resizing_visgen.hpp>
#include <stan/lang/generator/var_size_validating_visgen.hpp>
#include <stan/lang/generator/visgen.hpp>

#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_array_var_type.hpp>
#include <stan/lang/generator/generate_class_decl.hpp>
#include <stan/lang/generator/generate_class_decl_end.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_eigen_index_expression.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_include.hpp>
#include <stan/lang/generator/generate_includes.hpp>
#include <stan/lang/generator/generate_indexed_expr.hpp>
#include <stan/lang/generator/generate_indexed_expr_user.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_initialization.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_local_var_inits.hpp>
#include <stan/lang/generator/generate_member_var_decls.hpp>
#include <stan/lang/generator/generate_namespace_end.hpp>
#include <stan/lang/generator/generate_namespace_start.hpp>
#include <stan/lang/generator/generate_printable.hpp>
#include <stan/lang/generator/generate_private_decl.hpp>
#include <stan/lang/generator/generate_public_decl.hpp>
#include <stan/lang/generator/generate_quoted_expression.hpp>
#include <stan/lang/generator/generate_quoted_string.hpp>
#include <stan/lang/generator/generate_real_var_type.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/generate_typedef.hpp>
#include <stan/lang/generator/generate_typedefs.hpp>
#include <stan/lang/generator/generate_using.hpp>
#include <stan/lang/generator/generate_using_namespace.hpp>
#include <stan/lang/generator/generate_usings.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/generate_validate_transformed_params.hpp>
#include <stan/lang/generator/generate_validate_var_decl.hpp>
#include <stan/lang/generator/generate_validate_var_decls.hpp>
#include <stan/lang/generator/generate_var_resizing.hpp>
#include <stan/lang/generator/generate_version_comment.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>



#include <boost/variant/apply_visitor.hpp>
#include <boost/lexical_cast.hpp>

#include <stan/version.hpp>
#include <stan/lang/ast.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {

  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);
    void generate_expression(const expression& e, bool user_facing,
                             std::ostream& o);
    void generate_expression(const expression& e, bool user_facing,
                             bool is_var_context, std::ostream& o);
    void generate_bare_type(const expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& out);
    void generate_statement(const statement& s, int indent, std::ostream& o,
                            bool include_sampling, bool is_var_context,
                            bool is_fun_return);
    void generate_statement(const std::vector<statement>& ss, int indent,
                            std::ostream& o, bool include_sampling,
                            bool is_var_context, bool is_fun_return);
    void generate_idxs(const std::vector<idx>& idxs, std::ostream& o);
    void generate_idxs_user(const std::vector<idx>& idxs, std::ostream& o);





    struct is_numbered_statement_vis : public boost::static_visitor<bool> {
      bool operator()(const nil& st) const { return false; }
      bool operator()(const assignment& st) const { return true; }
      bool operator()(const assgn& st) const { return true; }
      bool operator()(const sample& st) const { return true; }
      bool operator()(const increment_log_prob_statement& t) const {
        return true;
      }
      bool operator()(const expression& st) const  { return true; }
      bool operator()(const statements& st) const  { return false; }
      bool operator()(const for_statement& st) const  { return true; }
      bool operator()(const conditional_statement& st) const { return true; }
      bool operator()(const while_statement& st) const { return true; }
      bool operator()(const break_continue_statement& st) const {
        return true;
      }
      bool operator()(const print_statement& st) const { return true; }
      bool operator()(const reject_statement& st) const { return true; }
      bool operator()(const no_op_statement& st) const { return true; }
      bool operator()(const return_statement& st) const { return true; }
    };

    void generate_statement(const statement& s,
                            int indent,
                            std::ostream& o,
                            bool include_sampling,
                            bool is_var_context,
                            bool is_fun_return) {
      is_numbered_statement_vis vis_is_numbered;
      if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  s.begin_line_ << ";"
          << EOL;
      }
      statement_visgen vis(indent, include_sampling, is_var_context,
                           is_fun_return, o);
      boost::apply_visitor(vis, s.statement_);
    }

    // FIXME:  don't ever call this -- call generate statement instead
    void generate_statements(const std::vector<statement>& ss,
                             int indent,
                             std::ostream& o,
                             bool include_sampling,
                             bool is_var_context,
                             bool is_fun_return) {
      statement_visgen vis(indent, include_sampling, is_var_context,
                           is_fun_return, o);
      for (size_t i = 0; i < ss.size(); ++i)
        boost::apply_visitor(vis, ss[i].statement_);
    }

    void generate_try(int indent,
                      std::ostream& o) {
      generate_indent(indent, o);
      o << "try {"
        << EOL;
    }

    void generate_catch_throw_located(int indent,
                                      std::ostream& o) {
      generate_indent(indent, o);
      o << "} catch (const std::exception& e) {"
        << EOL;
      generate_indent(indent + 1, o);
      o << "stan::lang::rethrow_located(e,current_statement_begin__);"
        << EOL;
      generate_comment("Next line prevents compiler griping about no return",
                       indent + 1, o);
      generate_indent(indent + 1, o);
      o << "throw std::runtime_error"
        << "(\"*** IF YOU SEE THIS, PLEASE REPORT A BUG ***\");"
        << EOL;
      generate_indent(indent, o);
      o << "}"
        << EOL;
    }

    void generate_located_statement(const statement& s,
                                    int indent,
                                    std::ostream& o,
                                    bool include_sampling,
                                    bool is_var_context,
                                    bool is_fun_return) {
      generate_try(indent, o);
      generate_statement(s, indent+1, o, include_sampling,
                         is_var_context, is_fun_return);
      generate_catch_throw_located(indent, o);
    }

    void generate_located_statements(const std::vector<statement>& ss,
                                     int indent,
                                     std::ostream& o,
                                     bool include_sampling,
                                     bool is_var_context,
                                     bool is_fun_return) {
      generate_try(indent, o);
      for (size_t i = 0; i < ss.size(); ++i)
        generate_statement(ss[i], indent + 1, o, include_sampling,
                           is_var_context, is_fun_return);
      generate_catch_throw_located(indent, o);
    }



    void generate_log_prob(const program& p,
                           std::ostream& o) {
      o << EOL;
      o << INDENT << "template <bool propto__, bool jacobian__, typename T__>"
        << EOL;
      o << INDENT << "T__ log_prob(vector<T__>& params_r__,"
        << EOL;
      o << INDENT << "             vector<int>& params_i__,"
        << EOL;
      o << INDENT << "             std::ostream* pstream__ = 0) const {"
        << EOL2;

      // use this dummy for inits
      o << INDENT2
        << "T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      o << INDENT2 << "T__ lp__(0.0);"
        << EOL;
      o << INDENT2 << "stan::math::accumulator<T__> lp_accum__;"
        << EOL2;

      bool is_var_context = true;
      bool is_fun_return = false;

      generate_comment("model parameters", 2, o);
      generate_local_var_inits(p.parameter_decl_, is_var_context, true, o);
      o << EOL;

      generate_comment("transformed parameters", 2, o);
      generate_local_var_decls(p.derived_decl_.first, 2, o, is_var_context,
                               is_fun_return);
      o << EOL;

      bool include_sampling = true;
      generate_located_statements(p.derived_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_validate_transformed_params(p.derived_decl_.first, 2, o);
      o << INDENT2
        << "const char* function__ = \"validate transformed params\";"
        << EOL;
      o << INDENT2
        << "(void) function__;  // dummy to suppress unused var warning"
        << EOL;

      generate_validate_var_decls(p.derived_decl_.first, 2, o);

      o << EOL;
      generate_comment("model body", 2, o);


      generate_located_statement(p.statement_, 2, o, include_sampling,
                                 is_var_context, is_fun_return);


      o << EOL;
      o << INDENT2 << "lp_accum__.add(lp__);" << EOL;
      o << INDENT2 << "return lp_accum__.sum();" << EOL2;
      o << INDENT << "} // log_prob()" << EOL2;

      o << INDENT
        << "template <bool propto, bool jacobian, typename T_>" << EOL;
      o << INDENT
        << "T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT << "           std::ostream* pstream = 0) const {" << EOL;
      o << INDENT << "  std::vector<T_> vec_params_r;" << EOL;
      o << INDENT << "  vec_params_r.reserve(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    vec_params_r.push_back(params_r(i));" << EOL;
      o << INDENT << "  std::vector<int> vec_params_i;" << EOL;
      o << INDENT
        << "  return log_prob<propto,jacobian,T_>(vec_params_r, "
        << "vec_params_i, pstream);" << EOL;
      o << INDENT << "}" << EOL2;
    }

    struct dump_member_var_visgen : public visgen {
      var_resizing_visgen var_resizer_;
      var_size_validating_visgen var_size_validator_;
      explicit dump_member_var_visgen(std::ostream& o)
        : visgen(o),
          var_resizer_(var_resizing_visgen(o)),
          var_size_validator_(var_size_validating_visgen(o,
                                                    "data initialization")) {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_i__ = context__.vals_i(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_"
             << dim << "__ < " << x.name_ << "_limit_" << dim
             << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_i__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim, o_);
          o_ << "}" << EOL;
        }
      }
      // minor changes to int_var_decl
      void operator()(const double_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim, o_);
          o_ << "}" << EOL;
        }
      }
      // extra outer loop around double_var_decl
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // change variable name from vector_var_decl
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // diff name of dims from vector
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT2 << "}" << EOL;
      }
      // extra loop and different accessor vs. vector
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        // FIXME: cut-and-paste of cov_matrix,
        //        very slightly different from matrix
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2
           << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_
           << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3
           << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_
           << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        // FIXME: cut and paste of cov_matrix
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;

        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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

        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        // FIXME: cut and paste of cholesky_factor_var_decl
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;

        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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

        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");"
           << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
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
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
    };

    void suppress_warning(const std::string& indent,
                          const std::string& var_name,
                          std::ostream& o) {
      o << indent << "(void) "
        << var_name << ";"
        << " // dummy call to supress warning"
        << EOL;
    }

    void generate_member_var_inits(const std::vector<var_decl>& vs,
                                   std::ostream& o) {
      dump_member_var_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

    void generate_destructor(const std::string& model_name,
                             std::ostream& o) {
      o << EOL
        << INDENT << "~" << model_name << "() { }"
        << EOL2;
    }

    // know all data is set and range expressions only depend on data
    struct set_param_ranges_visgen : public visgen {
      explicit set_param_ranges_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment_i(x.dims_);
        // for loop for ranges
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "for (size_t i_" << i << "__ = 0; ";
          o_ << "i_" << i << "__ < ";
          generate_expression(x.dims_[i], o_);
          o_ << "; ++i_" << i << "__) {" << EOL;
        }
        // add range
        generate_indent(x.dims_.size() + 2, o_);
        o_ << "param_ranges_i__.push_back(std::pair<int, int>(";
        generate_expression(x.range_.low_, o_);
        o_ << ", ";
        generate_expression(x.range_.high_, o_);
        o_ << "));" << EOL;
        // close for loop
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(x.dims_.size() + 1 - i, o_);
          o_ << "}" << EOL;
        }
      }
      void operator()(const double_var_decl& x) const {
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.M_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.N_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.M_, x.N_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const simplex_var_decl& x) const {
        // only K-1 vals
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << " - 1)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const ordered_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.K_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.K_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += ((";
        // N * (N + 1) / 2  +  (M - N) * M
        generate_expression(x.N_, o_);
        o_ << " * (";
        generate_expression(x.N_, o_);
        o_ << " + 1)) / 2 + (";
        generate_expression(x.M_, o_);
        o_ << " - ";
        generate_expression(x.N_, o_);
        o_ << ") * ";
        generate_expression(x.N_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        // FIXME: cut and paste ofcorr_matrix_var_decl
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        // (K * (K - 1))/2 + K  ?? define fun(K) = ??
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2 + ";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      // cut-and-paste from next for r
      void generate_increment_i(std::vector<expression> dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_i__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(std::vector<expression> dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_r__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(expression K,
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(K, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(expression M, expression N,
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(M, o_);
        o_ << " * ";
        generate_expression(N, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
    };

    /*
     * Generate statements in ctor_body which cumulatively determine
     * the size required for the vector of param ranges and the
     * range for each parameter in the model by iterating over the
     * list of parameter variable declarations
     */
    void generate_set_param_ranges(const std::vector<var_decl>& var_decls,
                                   std::ostream& o) {
      o << INDENT2 << "num_params_r__ = 0U;" << EOL;
      o << INDENT2 << "param_ranges_i__.clear();" << EOL;
      set_param_ranges_visgen vis(o);
      for (size_t i = 0; i < var_decls.size(); ++i)
        boost::apply_visitor(vis, var_decls[i].decl_);
    }

    void generate_constructor(const program& prog,
                              const std::string& model_name,
                              std::ostream& o) {
      // constructor without RNG or template parameter
      // FIXME(carpenter): remove this and only call full ctor
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "typedef boost::ecuyer1988 rng_t;" << EOL;
      o << INDENT2 << "rng_t base_rng(0);  // 0 seed default" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;

      // constructor with specified RNG
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    RNG& base_rng__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng__, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;

      // body of constructor now in function
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << "void ctor_body(stan::io::var_context& context__," << EOL;
      o << INDENT << "               RNG& base_rng__," << EOL;
      o << INDENT << "               std::ostream* pstream__) {" << EOL;
      o << INDENT2 << "current_statement_begin__ = -1;" << EOL2;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::" << model_name << "\";" << EOL;
      suppress_warning(INDENT2, "function__", o);
      o << INDENT2 << "size_t pos__;" << EOL;
      suppress_warning(INDENT2, "pos__", o);
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;

      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      o << INDENT2 << "// initialize member variables" << EOL;
      generate_member_var_inits(prog.data_decl_, o);

      o << EOL;
      generate_comment("validate, data variables", 2, o);
      generate_validate_var_decls(prog.data_decl_, 2, o);
      generate_comment("initialize data variables", 2, o);
      generate_var_resizing(prog.derived_data_decl_.first, o);
      o << EOL;

      bool include_sampling = false;
      bool is_var_context = false;
      bool is_fun_return = false;

      // need to fix generate_located_statements
      generate_located_statements(prog.derived_data_decl_.second,
                                  2, o, include_sampling, is_var_context,
                                  is_fun_return);

      o << EOL;
      generate_comment("validate transformed data", 2, o);
      generate_validate_var_decls(prog.derived_data_decl_.first, 2, o);

      o << EOL;
      generate_comment("validate, set parameter ranges", 2, o);
      generate_set_param_ranges(prog.parameter_decl_, o);
      // o << EOL << INDENT2 << "set_param_ranges();" << EOL;

      o << INDENT << "}" << EOL;
    }

    struct generate_init_visgen : public visgen {
      var_size_validating_visgen var_size_validator_;
      explicit generate_init_visgen(std::ostream& o)
        : visgen(o),
          var_size_validator_(o, "initialization") {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        generate_check_int(x.name_, x.dims_.size());
        var_size_validator_(x);
        generate_declaration(x.name_, "int", x.dims_, nil(), nil(), x.def_);
        generate_buffer_loop("i", x.name_, x.dims_);
        generate_write_loop("integer(", x.name_, x.dims_);
      }
      template <typename D>
      std::string function_args(const std::string& fun_prefix,
                                const D& x) const {
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

      void operator()(const double_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size());
        var_size_validator_(x);
        generate_declaration(x.name_, "double", x.dims_, nil(), nil(), x.def_);
        if (is_nil(x.def_)) {
          generate_buffer_loop("r", x.name_, x.dims_);
        }
        generate_write_loop(function_args("scalar", x),
                            x.name_, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.M_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_);
        generate_write_loop(function_args("vector", x),
                            x.name_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "row_vector_d", x.dims_, x.N_, nil(),
                             x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.N_);
        generate_write_loop(function_args("row_vector", x),
                            x.name_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop(function_args("matrix", x),
                            x.name_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("unit_vector_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("simplex_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("ordered_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("positive_ordered_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop("cholesky_factor_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cholesky_corr_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cov_matrix_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("corr_matrix_unconstrain(", x.name_, x.dims_);
      }
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
      void generate_name_dims(const std::string name,
                              size_t num_dims) const {
        o_ << name;
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "[i" << i << "__]";
      }
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
      void generate_indent_num_dims(size_t base_indent,
                                    const std::vector<expression>& dims,
                                    const expression& dim1,
                                    const expression& dim2) const {
        generate_indent(dims.size() + base_indent, o_);
        if (!is_nil(dim1)) o_ << INDENT;
        if (!is_nil(dim2)) o_ << INDENT;
      }
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
      void generate_check_int(const std::string& name, size_t /*n*/) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_i(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2 << "vals_i__ = context__.vals_i(\"" << name << "\");"
           << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
      void generate_check_double(const std::string& name, size_t /*n*/) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_r(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << name << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
    };


    void generate_init_method(const std::vector<var_decl>& vs,
                              std::ostream& o) {
      o << EOL;
      o << INDENT
        << "void transform_inits(const stan::io::var_context& context__,"
        << EOL;
      o << INDENT << "                     std::vector<int>& params_i__,"
        << EOL;
      o << INDENT << "                     std::vector<double>& params_r__,"
        << EOL;
      o << INDENT << "                     std::ostream* pstream__) const {"
        << EOL;
      o << INDENT2 << "stan::io::writer<double> "
        << "writer__(params_r__,params_i__);"
        << EOL;
      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "(void) pos__; // dummy call to supress warning" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;"
        << EOL;
      generate_init_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);

      o << EOL
        << INDENT2 << "params_r__ = writer__.data_r();" << EOL;
      o << INDENT2 << "params_i__ = writer__.data_i();" << EOL;
      o << INDENT << "}" << EOL2;

      o << INDENT
        << "void transform_inits(const stan::io::var_context& context," << EOL;
      o << INDENT
        << "                     "
        << "Eigen::Matrix<double,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT
        << "                     std::ostream* pstream__) const {" << EOL;
      o << INDENT << "  std::vector<double> params_r_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  transform_inits(context, params_i_vec, params_r_vec, pstream__);"
        << EOL;
      o << INDENT << "  params_r.resize(params_r_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r(i) = params_r_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

    struct write_dims_visgen : public visgen {
      explicit write_dims_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const  { }
      void operator()(const int_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void
      generate_dims_array(const std::vector<expression>& matrix_dims_exprs,
                          const std::vector<expression>& array_dims_exprs)
        const {
        o_ << INDENT2 << "dims__.resize(0);" << EOL;
        for (size_t i = 0; i < array_dims_exprs.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(array_dims_exprs[i].expr_, o_);
          o_ << ");" << EOL;
        }
        // cut and paste above with matrix_dims_exprs
        for (size_t i = 0; i < matrix_dims_exprs.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(matrix_dims_exprs[i].expr_, o_);
          o_ << ");" << EOL;
        }
        o_ << INDENT2 << "dimss__.push_back(dims__);" << EOL;
      }
    };

    void generate_dims_method(const program& prog,
                              std::ostream& o) {
      write_dims_visgen vis(o);
      o << EOL << INDENT
        << "void get_dims(std::vector<std::vector<size_t> >& dimss__) const {"
        << EOL;

      o << INDENT2 << "dimss__.resize(0);" << EOL;
      o << INDENT2 << "std::vector<size_t> dims__;" << EOL;

      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }
      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }
      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }
      o << INDENT << "}" << EOL2;
    }



    struct write_param_names_visgen : public visgen {
      explicit write_param_names_visgen(std::ostream& o)
        : visgen(o) {
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
      void
      generate_param_names(const std::string& name) const {
        o_ << INDENT2
           << "names__.push_back(\"" << name << "\");"
           << EOL;
      }
    };


    void generate_param_names_method(const program& prog,
                                     std::ostream& o) {
      write_param_names_visgen vis(o);
      o << EOL << INDENT
        << "void get_param_names(std::vector<std::string>& names__) const {"
        << EOL;

      o << INDENT2
        << "names__.resize(0);"
        << EOL;

      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }
      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }
      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }



    struct constrained_param_names_visgen : public visgen {
      explicit constrained_param_names_visgen(std::ostream& o)
        : visgen(o) {
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
    };


    void generate_constrained_param_names_method(const program& prog,
                                                 std::ostream& o) {
      o << EOL << INDENT
        << "void constrained_param_names("
        << "std::vector<std::string>& param_names__,"
        << EOL << INDENT
        << "                             bool include_tparams__ = true,"
        << EOL << INDENT
        << "                             bool include_gqs__ = true) const {"
        << EOL << INDENT2
        << "std::stringstream param_name_stream__;" << EOL;

      constrained_param_names_visgen vis(o);
      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__ && !include_tparams__) return;"
        << EOL;

      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__) return;"
        << EOL;

      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }

    struct unconstrained_param_names_visgen : public visgen {
      explicit unconstrained_param_names_visgen(std::ostream& o)
        : visgen(o) {
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
        matrix_args.push_back(binary_op(x.K_, "-", int_literal(1)));
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
        // FIXME: cut-and-paste of cov_matrix
        std::vector<expression> matrix_args;
        // (N * (N + 1)) / 2 + (M - N) * N
        matrix_args.push_back(binary_op(binary_op(binary_op(x.N_,
                                                            "*",
                                                            binary_op(x.N_,
                                                                      "+",
                                                               int_literal(1))),
                                                  "/",
                                                  int_literal(2)),
                                        "+",
                                        binary_op(binary_op(x.M_,
                                                            "-",
                                                            x.N_),
                                                  "*",
                                                  x.N_)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        // FIXME: cut-and-paste of corr_matrix
        std::vector<expression> matrix_args;
        // (K * (K - 1)) / 2
        matrix_args.push_back(binary_op(binary_op(x.K_,
                                                  "*",
                                                  binary_op(x.K_,
                                                            "-",
                                                            int_literal(1))),
                                        "/",
                                        int_literal(2)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(binary_op(x.K_,
                                        "+",
                                        binary_op(binary_op(x.K_,
                                                            "*",
                                                            binary_op(x.K_,
                                                                      "-",
                                                               int_literal(1))),
                               "/",
                               int_literal(2))));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(binary_op(binary_op(x.K_,
                                                  "*",
                                                  binary_op(x.K_,
                                                            "-",
                                                            int_literal(1))),
                                        "/",
                                        int_literal(2)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      // FIXME: sharing instead of cut-and-paste from constrained
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
    };


    void generate_unconstrained_param_names_method(const program& prog,
                                                   std::ostream& o) {
      o << EOL << INDENT
        << "void unconstrained_param_names("
        << "std::vector<std::string>& param_names__,"
        << EOL << INDENT
        << "                               bool include_tparams__ = true,"
        << EOL << INDENT
        << "                               bool include_gqs__ = true) const {"
        << EOL << INDENT2
        << "std::stringstream param_name_stream__;" << EOL;

      unconstrained_param_names_visgen vis(o);
      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__ && !include_tparams__) return;"
        << EOL;

      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__) return;"
        << EOL;

      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }


    // see init_member_var_visgen for cut & paste
    struct write_array_visgen : public visgen {
      explicit write_array_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int", "integer", EMPTY_EXP_VECTOR,
                                  x.name_, x.dims_);
      }
      // fixme -- reuse cut-and-pasted from other lub reader case
      template <typename D>
      void generate_initialize_array_bounded(const D& x,
                                             const std::string& base_type,
                                             const std::string& read_fun_prefix,
                                const std::vector<expression>& dim_args) const {
        std::vector<expression> read_args;
        std::string read_fun(read_fun_prefix);
        if (has_lub(x)) {
          read_fun += "_lub";
          read_args.push_back(x.range_.low_);
          read_args.push_back(x.range_.high_);
        } else if (has_lb(x)) {
          read_fun += "_lb";
          read_args.push_back(x.range_.low_);
        } else if (has_ub(x)) {
          read_fun += "_ub";
          read_args.push_back(x.range_.high_);
        }
        for (size_t i = 0; i < dim_args.size(); ++i)
          read_args.push_back(dim_args[i]);
        generate_initialize_array(base_type, read_fun, read_args,
                                  x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> read_args;
        generate_initialize_array_bounded(x, "double", "scalar", read_args);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array_bounded(x, "vector_d", "vector", read_args);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x, "row_vector_d", "row_vector",
                                          read_args);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x, "matrix_d", "matrix", read_args);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "unit_vector", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "simplex", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "ordered", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "positive_ordered", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array("matrix_d", "cholesky_factor", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "cholesky_corr", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "cov_matrix", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "corr_matrix", read_args,
                                  x.name_, x.dims_);
      }
      void generate_initialize_array(const std::string& var_type,
                                     const std::string& read_type,
                                     const std::vector<expression>& read_args,
                                     const std::string& name,
                                     const std::vector<expression>& dims) const{
        if (dims.size() == 0) {
          generate_indent(2, o_);
          o_ << var_type << " ";
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          o_ << ");" << EOL;
          return;
        }
        o_ << INDENT2;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
        o_ << var_type;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
        o_ << name << ";" << EOL;
        std::string name_dims(name);
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "size_t dim_"  << name << "_" << i << "__ = ";
          generate_expression(dims[i], o_);
          o_ << ";" << EOL;
          if (i < dims.size() - 1) {
            generate_indent(i + 2, o_);
            o_ << name_dims << ".resize(dim_" << name << "_" << i << "__);"
               << EOL;
            name_dims.append("[k_").append(to_string(i)).append("__]");
          }
          generate_indent(i + 2, o_);
          o_ << "for (size_t k_" << i << "__ = 0;"
             << " k_" << i << "__ < dim_" << name << "_" << i << "__;"
             << " ++k_" << i << "__) {" << EOL;
          if (i == dims.size() - 1) {
            generate_indent(i + 3, o_);
            o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
            for (size_t j = 0; j < read_args.size(); ++j) {
              if (j > 0) o_ << ",";
              generate_expression(read_args[j], o_);
            }
            o_ << "));" << EOL;
          }
        }

        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };




    struct write_array_vars_visgen : public visgen {
      explicit write_array_vars_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      // FIXME: template these out
      void operator()(const int_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }
      void operator()(const double_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.N_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void write_array(const std::string& name,
                       const std::vector<expression>& arraydims,
                       const std::vector<expression>& matdims) const {
        std::vector<expression> dims(arraydims);
        for (size_t i = 0; i < matdims.size(); ++i)
          dims.push_back(matdims[i]);

        if (dims.size() == 0) {
          o_ << INDENT2 << "vars__.push_back(" << name << ");" << EOL;
          return;
        }

        // for (size_t i = 0; i < dims.size(); ++i) {
        for (size_t i = dims.size(); i > 0; ) {
          --i;
          generate_indent((dims.size() - i) + 1, o_);
          o_ << "for (int k_" << i << "__ = 0;"
             << " k_" << i << "__ < ";
          generate_expression(dims[i], o_);
          o_ << "; ++k_" << i << "__) {" << EOL;
        }

        generate_indent(dims.size() + 2, o_);
        o_ << "vars__.push_back(" << name;
        if (arraydims.size() > 0) {
          o_ << '[';
          for (size_t i = 0; i < arraydims.size(); ++i) {
            if (i > 0) o_ << "][";
            o_ << "k_" << i << "__";
          }
          o_ << ']';
        }
        if (matdims.size() > 0) {
          o_ << "(k_" << arraydims.size() << "__";
          if (matdims.size() > 1)
            o_ << ", k_" << (arraydims.size() + 1) << "__";
          o_ << ")";
        }
        o_ << ");" << EOL;

        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };


    void generate_write_array_method(const program& prog,
                                     const std::string& model_name,
                                     std::ostream& o) {
      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng__," << EOL;
      o << INDENT << "                 std::vector<double>& params_r__," << EOL;
      o << INDENT << "                 std::vector<int>& params_i__," << EOL;
      o << INDENT << "                 std::vector<double>& vars__," << EOL;
      o << INDENT << "                 bool include_tparams__ = true," << EOL;
      o << INDENT << "                 bool include_gqs__ = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream__ = 0) const {" << EOL;
      o << INDENT2 << "vars__.resize(0);" << EOL;
      o << INDENT2
        << "stan::io::reader<double> in__(params_r__,params_i__);"<< EOL;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::write_array\";" << EOL;
      suppress_warning(INDENT2, "function__", o);

      // declares, reads, and sets parameters
      generate_comment("read-transform, write parameters", 2, o);
      write_array_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);

      // this is for all other values
      write_array_vars_visgen vis_writer(o);

      // writes parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis_writer, prog.parameter_decl_[i].decl_);
      o << EOL;

      o << INDENT2 << "if (!include_tparams__) return;"
        << EOL;
      generate_comment("declare and define transformed parameters", 2, o);
      o << INDENT2 <<  "double lp__ = 0.0;" << EOL;
      suppress_warning(INDENT2, "lp__", o);
      o << INDENT2 << "stan::math::accumulator<double> lp_accum__;" << EOL2;

      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      bool is_var_context = false;
      bool is_fun_return = false;
      generate_local_var_decls(prog.derived_decl_.first, 2, o, is_var_context,
                               is_fun_return);
      o << EOL;
      bool include_sampling = false;
      generate_located_statements(prog.derived_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_comment("validate transformed parameters", 2, o);
      generate_validate_var_decls(prog.derived_decl_.first, 2, o);
      o << EOL;

      generate_comment("write transformed parameters", 2, o);
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.derived_decl_.first[i].decl_);
      o << EOL;

      o << INDENT2 << "if (!include_gqs__) return;"
        << EOL;
      generate_comment("declare and define generated quantities", 2, o);
      generate_local_var_decls(prog.generated_decl_.first, 2, o,
                               is_var_context, is_fun_return);

      o << EOL;
      generate_located_statements(prog.generated_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_comment("validate generated quantities", 2, o);
      generate_validate_var_decls(prog.generated_decl_.first, 2, o);
      o << EOL;

      generate_comment("write generated quantities", 2, o);
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.generated_decl_.first[i].decl_);
      if (prog.generated_decl_.first.size() > 0)
        o << EOL;

      o << INDENT << "}" << EOL2;

      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng," << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,"
        << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& vars,"
        << EOL;
      o << INDENT << "                 bool include_tparams = true," << EOL;
      o << INDENT << "                 bool include_gqs = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream = 0) const {" << EOL;
      o << INDENT
        << "  std::vector<double> params_r_vec(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r_vec[i] = params_r(i);" << EOL;
      o << INDENT << "  std::vector<double> vars_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  write_array(base_rng,params_r_vec,params_i_vec,"
        << "vars_vec,include_tparams,include_gqs,pstream);" << EOL;
      o << INDENT << "  vars.resize(vars_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < vars.size(); ++i)" << EOL;
      o << INDENT << "    vars(i) = vars_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

    void generate_model_name_method(const std::string& model_name,
                                    std::ostream& out) {
      out << INDENT << "static std::string model_name() {" << EOL
          << INDENT2 << "return \"" << model_name << "\";" << EOL
          << INDENT << "}" << EOL2;
    }

    void generate_model_typedef(const std::string& model_name,
                                std::ostream& out) {
      out << "typedef " << model_name << "_namespace::" << model_name
          << " stan_model;" <<EOL2;
    }

    void generate_bare_type(const expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& out) {
      for (size_t d = 0; d < t.num_dims_; ++d)
        out << "std::vector<";

      bool is_template_type = false;
      switch (t.base_type_) {
      case INT_T :
        out << "int";
        is_template_type = false;
        break;
      case DOUBLE_T:
        out << scalar_t_name;
        is_template_type = false;
        break;
      case VECTOR_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", Eigen::Dynamic,1>";
        is_template_type = true;
        break;
      case ROW_VECTOR_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", 1,Eigen::Dynamic>";
        is_template_type = true;
        break;
      case MATRIX_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", Eigen::Dynamic,Eigen::Dynamic>";
        is_template_type = true;
        break;
      case VOID_T:
        out << "void";
        break;
      default:
        out << "UNKNOWN TYPE";
      }

      for (size_t d = 0; d < t.num_dims_; ++d) {
        if (d > 0 || is_template_type)
          out << " ";
        out << ">";
      }
    }

    void generate_arg_decl(bool gen_const,
                           bool gen_ref,
                           const arg_decl& decl,
                           const std::string& scalar_t_name,
                           std::ostream& out) {
      if (gen_const)
        out << "const ";
      generate_bare_type(decl.arg_type_, scalar_t_name, out);
      if (gen_ref)
        out << "&";
      out << " " << decl.name_;
    }

    bool has_only_int_args(const function_decl_def& fun) {
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i)
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T)
          return false;
      return true;
    }

    std::string fun_scalar_type(const function_decl_def& fun,
                                bool is_lp) {
      size_t num_args = fun.arg_decls_.size();
      // nullary, non-lp
      if (has_only_int_args(fun) && !is_lp)
        return "double";

      // need template metaprogram to construct return
      std::stringstream ss;
      ss << "typename boost::math::tools::promote_args<";
      int num_open_brackets = 1;
      int num_generated_params = 0;
      for (size_t i = 0; i < num_args; ++i) {
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
          // two conditionals cut and pasted below
          if (num_generated_params > 0)
            ss << ", ";
          if (num_generated_params == 4) {
            ss << "typename boost::math::tools::promote_args<";
            num_generated_params = 0;
            ++num_open_brackets;
          }
          ss << "T" << i << "__";
          ++num_generated_params;
        }
      }
      if (is_lp) {
        if (num_generated_params > 0)
          ss << ", ";
        // set threshold at 4 so always room for one more param at end
        ss << "T_lp__";
      }
      for (int i = 0; i < num_open_brackets; ++i)
        ss << ">::type";
      return ss.str();
    }
    // copy/modify for conditional_op???


    bool needs_template_params(const function_decl_def& fun) {
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
          return true;
        }
      }
      return false;
    }


    void generate_function_template_parameters(const function_decl_def& fun,
                                               bool is_rng,
                                               bool is_lp,
                                               bool is_log,
                                               std::ostream& out) {
      if (needs_template_params(fun)) {
        out << "template <";
        bool continuing_tps = false;
        if (is_log) {
          out << "bool propto";
          continuing_tps = true;
        }
        for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
          // no template parameter for int-based args
          if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
            if (continuing_tps)
              out << ", ";
            out << "typename T" << i << "__";
            continuing_tps = true;
          }
        }
        if (is_rng) {
          if (continuing_tps)
            out << ", ";
          out << "class RNG";
          continuing_tps = true;
        } else if (is_lp) {
          if (continuing_tps)
            out << ", ";
          out << "typename T_lp__, typename T_lp_accum__";
          continuing_tps = true;
        }
        out << ">" << EOL;
      } else {  // no-arg function
        if (is_rng) {
          // nullary RNG case
          out << "template <class RNG>" << EOL;
        } else if (is_lp) {
          out << "template <typename T_lp__, typename T_lp_accum__>"
              << EOL;
        } else if (is_log) {
          out << "template <bool propto>"
              << EOL;
        }
      }
    }

    void generate_function_inline_return_type(const function_decl_def& fun,
                                              const std::string& scalar_t_name,
                                              int indent,
                                              std::ostream& out) {
      generate_indent(indent, out);
      generate_bare_type(fun.return_type_, scalar_t_name, out);
      out << EOL;
    }

    void generate_function_name(const function_decl_def& fun,
                                std::ostream& out) {
      out << fun.name_;
    }


    void generate_function_arguments(const function_decl_def& fun,
                                     bool is_rng,
                                     bool is_lp,
                                     bool is_log,
                                     std::ostream& out) {
      // arguments
      out << "(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        std::string template_type_i
          = "T" + boost::lexical_cast<std::string>(i) + "__";
        generate_arg_decl(true, true, fun.arg_decls_[i], template_type_i, out);
        if (i + 1 < fun.arg_decls_.size()) {
          out << "," << EOL << INDENT;
          for (size_t i = 0; i <= fun.name_.size(); ++i)
            out << " ";
        }
      }
      if ((is_rng || is_lp) && fun.arg_decls_.size() > 0)
        out << ", ";
      if (is_rng)
        out << "RNG& base_rng__";
      else if (is_lp)
        out << "T_lp__& lp__, T_lp_accum__& lp_accum__";
      if (is_rng || is_lp || fun.arg_decls_.size() > 0)
        out << ", ";
      out << "std::ostream* pstream__";
      out << ")";
    }

    void generate_functor_arguments(const function_decl_def& fun,
                                    bool is_rng,
                                    bool is_lp,
                                    bool is_log,
                                    std::ostream& out) {
      // arguments
      out << "(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (i > 0)
          out << ", ";
        out << fun.arg_decls_[i].name_;
      }
      if ((is_rng || is_lp) && fun.arg_decls_.size() > 0)
        out << ", ";
      if (is_rng)
        out << "base_rng__";
      else if (is_lp)
        out << "lp__, lp_accum__";
      if (is_rng || is_lp || fun.arg_decls_.size() > 0)
        out << ", ";
      out << "pstream__";
      out << ")";
    }



    void generate_function_body(const function_decl_def& fun,
                                const std::string& scalar_t_name,
                                std::ostream& out) {
      // no-op body
      if (fun.body_.is_no_op_statement()) {
        out << ";" << EOL;
        return;
      }
      out << " {" << EOL;
      out << INDENT
          << "typedef " << scalar_t_name << " fun_scalar_t__;"
          << EOL;
      out << INDENT
          << "typedef "
          << ((fun.return_type_.base_type_ == INT_T)
              ? "int" : "fun_scalar_t__")
          << " fun_return_scalar_t__;"
          << EOL;
      out << INDENT
          << "const static bool propto__ = true;"
          << EOL
          << INDENT
          << "(void) propto__;"
          << EOL;
      // use this dummy for inits
      out << INDENT2
          << "fun_scalar_t__ "
          << "DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      out << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;
      bool is_var_context = false;
      bool is_fun_return = true;
      bool include_sampling = true;
      out << INDENT
          << "int current_statement_begin__ = -1;"
          << EOL;

      generate_located_statement(fun.body_, 1, out,
                                 include_sampling, is_var_context,
                                 is_fun_return);

      out << "}"
          << EOL;
    }
    void generate_propto_default_function_body(const function_decl_def& fun,
                                               std::ostream& out) {
      out << " {" << EOL;
      out << INDENT << "return ";
      out << fun.name_ << "<false>(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (i > 0)
          out << ",";
        out << fun.arg_decls_[i].name_;
      }
      if (fun.arg_decls_.size() > 0)
        out << ", ";
      out << "pstream__";
      out << ");" << EOL;
      out << "}" << EOL;
    }

    void generate_propto_default_function(const function_decl_def& fun,
                                          const std::string& scalar_t_name,
                                          std::ostream& out) {
      generate_function_template_parameters(fun, false, false, false, out);
      generate_function_inline_return_type(fun, scalar_t_name, 0, out);
      generate_function_name(fun, out);
      generate_function_arguments(fun, false, false, false, out);
      generate_propto_default_function_body(fun, out);
    }

    /**
     * Generate the specified function and optionally its default for
     * propto=false for functions ending in _log.
     *
     * Exact behavior differs for unmarked functions, and functions
     * ending in one of "_rng", "_lp", or "_log".
     *
     * @param[in] fun function AST object
     * @param[in, out] out output stream to which function definition
     * is written
     */
    void generate_function(const function_decl_def& fun,
                           std::ostream& out) {
      bool is_rng = ends_with("_rng", fun.name_);
      bool is_lp = ends_with("_lp", fun.name_);
      bool is_pf = ends_with("_log", fun.name_)
        || ends_with("_lpdf", fun.name_) || ends_with("_lpmf", fun.name_);
      std::string scalar_t_name = fun_scalar_type(fun, is_lp);

      generate_function_template_parameters(fun, is_rng, is_lp, is_pf, out);
      generate_function_inline_return_type(fun, scalar_t_name, 0, out);
      generate_function_name(fun, out);
      generate_function_arguments(fun, is_rng, is_lp, is_pf, out);
      generate_function_body(fun, scalar_t_name, out);

      // need a second function def for default propto=false for _log
      // funs; but don't want duplicate def, so don't do it for
      // forward decl when body is no-op
      if (is_pf && !fun.body_.is_no_op_statement())
        generate_propto_default_function(fun, scalar_t_name, out);
      out << EOL;
    }

    void generate_function_functor(const function_decl_def& fun,
                                   std::ostream& out) {
      if (fun.body_.is_no_op_statement())
        return;   // forward declaration, so no functor needed

      bool is_rng = ends_with("_rng", fun.name_);
      bool is_lp = ends_with("_lp", fun.name_);
      bool is_pf = ends_with("_log", fun.name_)
        || ends_with("_lpdf", fun.name_) || ends_with("_lpmf", fun.name_);
      std::string scalar_t_name = fun_scalar_type(fun, is_lp);

      out << EOL << "struct ";
      generate_function_name(fun, out);
      out << "_functor__ {" << EOL;

      out << INDENT;
      generate_function_template_parameters(fun, is_rng, is_lp, is_pf, out);

      out << INDENT;
      generate_function_inline_return_type(fun, scalar_t_name, 1, out);

      out <<  INDENT << "operator()";
      generate_function_arguments(fun, is_rng, is_lp, is_pf, out);
      out << " const {" << EOL;

      out << INDENT2 << "return ";
      generate_function_name(fun, out);
      generate_functor_arguments(fun, is_rng, is_lp, is_pf, out);
      out << ";" << EOL;
      out << INDENT << "}"  << EOL;
      out << "};" << EOL2;
    }


    void generate_functions(const std::vector<function_decl_def>& funs,
                            std::ostream& out) {
      for (size_t i = 0; i < funs.size(); ++i) {
        generate_function(funs[i], out);
        generate_function_functor(funs[i], out);
      }
    }

    void generate_member_var_decls_all(const program& prog,
                                       std::ostream& out) {
      generate_member_var_decls(prog.data_decl_, 1, out);
      generate_member_var_decls(prog.derived_data_decl_.first, 1, out);
    }

    void generate_globals(std::ostream& out) {
      out << "static int current_statement_begin__;"
          << EOL2;
    }


    void generate_cpp(const program& prog,
                      const std::string& model_name,
                      std::ostream& out) {
      generate_version_comment(out);
      generate_includes(out);
      generate_namespace_start(model_name, out);
      generate_usings(out);
      generate_typedefs(out);
      generate_globals(out);
      generate_functions(prog.function_decl_defs_, out);
      generate_class_decl(model_name, out);
      generate_private_decl(out);
      generate_member_var_decls_all(prog, out);
      generate_public_decl(out);
      generate_constructor(prog, model_name, out);
      generate_destructor(model_name, out);
      // put back if ever need integer params
      // generate_set_param_ranges(prog.parameter_decl_, out);
      generate_init_method(prog.parameter_decl_, out);
      generate_log_prob(prog, out);
      generate_param_names_method(prog, out);
      generate_dims_method(prog, out);
      generate_write_array_method(prog, model_name, out);
      generate_model_name_method(model_name, out);
      generate_constrained_param_names_method(prog, out);
      generate_unconstrained_param_names_method(prog, out);
      generate_class_decl_end(out);
      generate_namespace_end(out);
      generate_model_typedef(model_name, out);
    }

  }

}

#endif
