#ifndef STAN_LANG_GENERATOR_HPP
#define STAN_LANG_GENERATOR_HPP


// FIXME(carpenter): move into AST
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>

// FIXME(carpenter): move into general utilities
#include <stan/lang/generator/to_string.hpp>

#include <stan/lang/generator/is_numbered_statement_vis.hpp>

#include <stan/lang/generator/constrained_param_names_visgen.hpp>
#include <stan/lang/generator/dump_member_var_visgen.hpp>
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/printable_visgen.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>
#include <stan/lang/generator/init_local_var_visgen.hpp>
#include <stan/lang/generator/init_vars_visgen.hpp>
#include <stan/lang/generator/init_visgen.hpp>
#include <stan/lang/generator/local_var_decl_visgen.hpp>
#include <stan/lang/generator/local_var_init_nan_visgen.hpp>
#include <stan/lang/generator/member_var_decl_visgen.hpp>
#include <stan/lang/generator/set_param_ranges_visgen.hpp>
#include <stan/lang/generator/statement_visgen.hpp>
#include <stan/lang/generator/unconstrained_param_names_visgen.hpp>
#include <stan/lang/generator/validate_var_decl_visgen.hpp>
#include <stan/lang/generator/validate_transformed_params_visgen.hpp>
#include <stan/lang/generator/var_resizing_visgen.hpp>
#include <stan/lang/generator/var_size_validating_visgen.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <stan/lang/generator/write_array_visgen.hpp>
#include <stan/lang/generator/write_array_vars_visgen.hpp>
#include <stan/lang/generator/write_dims_visgen.hpp>
#include <stan/lang/generator/write_param_names_visgen.hpp>

#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_array_var_type.hpp>
#include <stan/lang/generator/generate_catch_throw_located.hpp>
#include <stan/lang/generator/generate_class_decl.hpp>
#include <stan/lang/generator/generate_class_decl_end.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_constrained_param_names_method.hpp>
#include <stan/lang/generator/generate_constructor.hpp>
#include <stan/lang/generator/generate_destructor.hpp>
#include <stan/lang/generator/generate_dims_method.hpp>
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
#include <stan/lang/generator/generate_init_method.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_initialization.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_local_var_inits.hpp>
#include <stan/lang/generator/generate_located_statement.hpp>
#include <stan/lang/generator/generate_located_statements.hpp>
#include <stan/lang/generator/generate_log_prob.hpp>
#include <stan/lang/generator/generate_member_var_decls.hpp>
#include <stan/lang/generator/generate_member_var_inits.hpp>
#include <stan/lang/generator/generate_model_name_method.hpp>
#include <stan/lang/generator/generate_model_typedef.hpp>
#include <stan/lang/generator/generate_namespace_end.hpp>
#include <stan/lang/generator/generate_namespace_start.hpp>
#include <stan/lang/generator/generate_param_names_method.hpp>
#include <stan/lang/generator/generate_printable.hpp>
#include <stan/lang/generator/generate_private_decl.hpp>
#include <stan/lang/generator/generate_public_decl.hpp>
#include <stan/lang/generator/generate_quoted_expression.hpp>
#include <stan/lang/generator/generate_quoted_string.hpp>
#include <stan/lang/generator/generate_real_var_type.hpp>
#include <stan/lang/generator/generate_set_param_ranges.hpp>
#include <stan/lang/generator/generate_statement.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/generate_typedef.hpp>
#include <stan/lang/generator/generate_typedefs.hpp>
#include <stan/lang/generator/generate_try.hpp>
#include <stan/lang/generator/generate_unconstrained_param_names_method.hpp>
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
#include <stan/lang/generator/generate_write_array_method.hpp>

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
