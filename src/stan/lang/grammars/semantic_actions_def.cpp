#ifndef STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_DEF_CPP
#define STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_DEF_CPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <cstddef>
#include <limits>
#include <climits>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stan {

  namespace lang {

    /**
     * Set original name of specified function to name and add
     * "stan::math::" namespace qualifier to name.
     *
     * @param[in, out] f Function to qualify.
     */
    void qualify(fun& f) {
      f.original_name_ = f.name_;
      f.name_ = "stan::math::" + f.name_;
    }

    /**
     * Add qualifier "stan::math::" to nullary functions defined in
     * the Stan language.  Sets original name of specified function to
     * name and add "stan::math::" namespace qualifier to name.
     *
     * @param[in, out] f Function to qualify.
     */
    void qualify_builtins(fun& f) {
      if (f.args_.size() > 0) return;
      if (f.name_ == "e" || f.name_ == "pi" || f.name_ == "log2"
          || f.name_ == "log10" || f.name_ == "sqrt2"
          || f.name_ == "not_a_number" || f.name_ == "positive_infinity"
          || f.name_ == "negative_infinity" || f.name_ == "machine_precision")
        qualify(f);
    }

    /**
     * Add namespace qualifier stan::math:: to specify Stan versions
     * of functions to avoid ambiguities with versions defined in
     * math.h in the top-level namespace.  Sets original name of
     * specified function to name and add <code>stan::math::</code>
     * namespace qualifier to name.
     *
     * @param[in, out] f Function to qualify.
     */
    void qualify_cpp11_builtins(fun& f) {
      if (f.args_.size() == 1
          && (f.name_ == "acosh"|| f.name_ == "asinh" || f.name_ == "atanh"
              || f.name_ == "exp2" || f.name_ == "expm1" || f.name_ == "log1p"
              || f.name_ == "log2" || f.name_ == "cbrt" || f.name_ == "erf"
              || f.name_ == "erfc" || f.name_ == "tgamma" || f.name_ == "lgamma"
              || f.name_ == "round" || f.name_ == "trunc"))
          qualify(f);
      else if (f.args_.size() == 2
               && (f.name_ == "fdim" || f.name_ == "fmax" || f.name_ == "fmin"
                   || f.name_ == "hypot"))
        qualify(f);
      else if (f.args_.size() == 3 && f.name_ == "fma")
        qualify(f);
    }

    bool has_prob_suffix(const std::string& s) {
      return ends_with("_lpdf", s) || ends_with("_lpmf", s)
        || ends_with("_lcdf", s) || ends_with("_lccdf", s);
    }

    void replace_suffix(const std::string& old_suffix,
                        const std::string& new_suffix, fun& f) {
      if (!ends_with(old_suffix, f.name_)) return;
      f.original_name_ = f.name_;
      f.name_ = f.name_.substr(0, f.name_.size() - old_suffix.size())
        + new_suffix;
    }

    bool deprecate_fun(const std::string& old_name, const std::string& new_name,
                       fun& f, std::ostream& msgs) {
      if (f.name_ != old_name) return false;
      f.original_name_ = f.name_;
      f.name_ = new_name;
      msgs << "Warning: Function name '" << old_name << "' is deprecated"
           << " and will be removed in a later release; please replace"
           << " with '" << new_name << "'" << std::endl;
      return true;
    }

    bool deprecate_suffix(const std::string& deprecated_suffix,
                          const std::string& replacement, fun& f,
                          std::ostream& msgs) {
      if (!ends_with(deprecated_suffix, f.name_)) return false;
      msgs << "Warning: Deprecated function '" << f.name_ << "';"
           << " please replace suffix '" << deprecated_suffix
           << "' with " << replacement << std::endl;
      return true;
    }

    void set_fun_type(fun& fun, std::ostream& error_msgs) {
      std::vector<expr_type> arg_types;
      for (size_t i = 0; i < fun.args_.size(); ++i)
        arg_types.push_back(fun.args_[i].expression_type());
      fun.type_ = function_signatures::instance()
        .get_result_type(fun.name_, arg_types, error_msgs);
    }

    int num_dimss(std::vector<std::vector<stan::lang::expression> >& dimss) {
      int sum = 0;
      for (size_t i = 0; i < dimss.size(); ++i)
        sum += dimss[i].size();
      return sum;
    }

    bool is_double_return(const std::string& function_name,
                          const std::vector<expr_type>& arg_types,
                          std::ostream& error_msgs) {
      return function_signatures::instance()
        .get_result_type(function_name, arg_types, error_msgs, true)
        .is_primitive_double();
    }

    bool is_univariate(const expr_type& et) {
      return et.num_dims_ == 0
        && (et.base_type_ == INT_T
            || et.base_type_ == DOUBLE_T);
    }

    bool can_assign_to_lhs_var(const std::string& lhs_var_name,
                               const scope& var_scope,
                               const variable_map& vm,
                               std::ostream& error_msgs) {
      // validate scope matches declaration scope
      scope lhs_origin = vm.get_scope(lhs_var_name);
      if (lhs_origin.program_block() != var_scope.program_block()) {
        error_msgs << "Cannot assign to variable outside of declaration block"
                   << "; left-hand-side variable origin=";
        print_scope(error_msgs, lhs_origin);
        error_msgs << std::endl;
        return false;
      }
      // enforce constancy of function args
      if (!lhs_origin.is_local() && lhs_origin.fun()) {
        error_msgs << "Cannot assign to function argument variables."
                   << std::endl
                   << "Use local variables instead."
                   << std::endl;
        return false;
      }
      return true;
    }

    expr_type infer_var_dims_type(const base_var_decl& var_type,
                                  const variable_dims& var_dims) {
      size_t var_num_dims = var_type.dims_.size();
      size_t num_index_dims = var_dims.dims_.size();
      return infer_type_indexing(var_type.base_type_,
                                 var_num_dims,
                                 num_index_dims);
    }

    bool has_same_shape(const expr_type& lhs_type,
                        const expression& rhs_expr,
                        const std::string& name,
                        const std::string& stmt_type,
                        std::ostream& error_msgs) {
      if (lhs_type.num_dims_ != rhs_expr.expression_type().num_dims_) {
        error_msgs << "Dimension mismatch in "
                   << stmt_type
                   << "; variable name = "
                   << name
                   << ", num dimensions given = "
                   << lhs_type.num_dims_
                   << "; right-hand side dimensions = "
                   << rhs_expr.expression_type().num_dims_
                   << std::endl;
        return false;
      }
      base_expr_type lhs_base_type = lhs_type.base_type_;
      base_expr_type rhs_base_type = rhs_expr.expression_type().base_type_;
      // allow int -> double promotion, even in arrays
      bool types_compatible =
        (lhs_base_type == rhs_base_type
         || (lhs_base_type == DOUBLE_T && rhs_base_type == INT_T));
      if (!types_compatible) {
        error_msgs << "Base type mismatch in "
                   << stmt_type
                   << "; variable name = "
                   << name
                   << ", type = ";
        write_base_expr_type(error_msgs, lhs_base_type);
        error_msgs << "; right-hand side type=";
        write_base_expr_type(error_msgs, rhs_base_type);
        error_msgs << std::endl;
        return false;
      }
      return true;
    }


    // //////////////////////////////////
    // *** functors for grammar rules ***
    // //////////////////////////////////

    void validate_double_expr::operator()(const expression& expr,
                              bool& pass,
                              std::stringstream& error_msgs)
      const {
      if (!expr.expression_type().is_primitive_double()
          && !expr.expression_type().is_primitive_int()) {
        error_msgs << "expression denoting real required; found type="
                   << expr.expression_type() << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_double_expr> validate_double_expr_f;

    template <typename L, typename R>
    void assign_lhs::operator()(L& lhs, const R& rhs) const {
      lhs = rhs;
    }
    boost::phoenix::function<assign_lhs> assign_lhs_f;

    template void assign_lhs::operator()(expression&, const expression&) const;
    template void assign_lhs::operator()(expression&, const double_literal&)
      const;
    template void assign_lhs::operator()(expression&, const int_literal&) const;
    template void assign_lhs::operator()(expression&, const integrate_ode&)
      const;
    template void assign_lhs::operator()(expression&,
                                         const integrate_ode_control&)
      const;
    template void assign_lhs::operator()(expression&, const algebra_solver&)
      const;
    template void assign_lhs::operator()(expression&,
                                         const algebra_solver_control&)
      const;
    template void assign_lhs::operator()(array_expr&,
                                         const array_expr&) const;
    template void assign_lhs::operator()(matrix_expr&,
                                         const matrix_expr&) const;
    template void assign_lhs::operator()(row_vector_expr&,
                                         const row_vector_expr&) const;
    template void assign_lhs::operator()(int&, const int&) const;
    template void assign_lhs::operator()(size_t&, const size_t&) const;
    template void assign_lhs::operator()(statement&, const statement&) const;
    template void assign_lhs::operator()(std::vector<var_decl>&,
                                         const std::vector<var_decl>&) const;
    template void assign_lhs::operator()(std::vector<idx>&,
                                         const std::vector<idx>&) const;
    template void assign_lhs::operator()(
                         std::vector<std::vector<expression> >&,
                         const std::vector<std::vector<expression> >&) const;
    template void assign_lhs::operator()(fun&, const fun&) const;
    template void assign_lhs::operator()(variable&, const variable&) const;

    void validate_expr_type3::operator()(const expression& expr, bool& pass,
                                         std::ostream& error_msgs) const {
      pass = !expr.expression_type().is_ill_formed();
      if (!pass)
        error_msgs << "expression is ill formed" << std::endl;
    }
    boost::phoenix::function<validate_expr_type3> validate_expr_type3_f;


    void is_prob_fun::operator()(const std::string& s,
                                 bool& pass) const {
      pass = has_prob_suffix(s);
    }
    boost::phoenix::function<is_prob_fun> is_prob_fun_f;

    void addition_expr3::operator()(expression& expr1, const expression& expr2,
                                    std::ostream& error_msgs) const {
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()) {
        expr1 += expr2;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("add", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<addition_expr3> addition3_f;

    void subtraction_expr3::operator()(expression& expr1,
                                       const expression& expr2,
                                       std::ostream& error_msgs) const {
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()) {
        expr1 -= expr2;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("subtract", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<subtraction_expr3> subtraction3_f;

    void increment_size_t::operator()(size_t& lhs) const {
      ++lhs;
    }
    boost::phoenix::function<increment_size_t> increment_size_t_f;


    void validate_conditional_op::operator()(conditional_op& conditional_op,
                                             const scope& var_scope,
                                             bool& pass,
                                             const variable_map& var_map,
                                             std::ostream& error_msgs) const {
      expr_type cond_type = conditional_op.cond_.expression_type();
      if (!cond_type.is_primitive_int()) {
        error_msgs << "condition in ternary expression must be"
                   << " primitive int or real;"
                   << " found type=" << cond_type
                   << std::endl;
        pass = false;
        return;
      }

      expr_type true_val_type = conditional_op.true_val_.expression_type();
      base_expr_type true_val_base_type = true_val_type.base_type_;
      expr_type false_val_type = conditional_op.false_val_.expression_type();
      base_expr_type false_val_base_type = false_val_type.base_type_;
      bool types_compatible
        = (true_val_type == false_val_type)
        || (true_val_type.is_primitive() && false_val_type.is_primitive()
           && (true_val_base_type == false_val_base_type
               || (true_val_base_type == DOUBLE_T
                   && false_val_base_type == INT_T)
               || (true_val_base_type == INT_T
                   && false_val_base_type == DOUBLE_T)));

      if (!types_compatible) {
        error_msgs << "base type mismatch in ternary expression,"
                   << " expression when true is: ";
        write_base_expr_type(error_msgs, true_val_base_type);
        error_msgs << "; expression when false is: ";
        write_base_expr_type(error_msgs, false_val_base_type);
        error_msgs << std::endl;
        pass = false;
        return;
      }

      if (true_val_type.is_primitive())
        conditional_op.type_
          = (true_val_base_type == false_val_base_type)
          ? true_val_base_type
          : DOUBLE_T;
      else
        conditional_op.type_ = true_val_type;

      conditional_op.has_var_ = has_var(conditional_op, var_map);
      conditional_op.scope_ = var_scope;
      pass = true;
    }
    boost::phoenix::function<validate_conditional_op>
    validate_conditional_op_f;

    void binary_op_expr::operator()(expression& expr1, const expression& expr2,
                                    const std::string& op,
                                    const std::string& fun_name,
                                    std::ostream& error_msgs) const {
      if (!expr1.expression_type().is_primitive()
          || !expr2.expression_type().is_primitive()) {
        error_msgs << "binary infix operator " << op
                   << " with functional interpretation " << fun_name
                   << " requires arguments or primitive type (int or real)"
                   << ", found left type=" << expr1.expression_type()
                   << ", right arg type=" << expr2.expression_type()
                   << "; "
                   << std::endl;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f(fun_name, args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<binary_op_expr> binary_op_f;

    void validate_non_void_arg_function::operator()(const expr_type& arg_type,
                                            bool& pass,
                                            std::ostream& error_msgs) const {
      pass = !arg_type.is_void();
      if (!pass)
        error_msgs << "Functions cannot contain void argument types; "
                   << "found void argument."
                   << std::endl;
    }
    boost::phoenix::function<validate_non_void_arg_function>
    validate_non_void_arg_f;

    void set_void_function:: operator()(const expr_type& return_type,
                                        scope& var_scope, bool& pass,
                                        std::ostream& error_msgs) const {
      if (return_type.is_void() && return_type.num_dims() > 0) {
        error_msgs << "Void return type may not have dimensions declared."
                   << std::endl;
        pass = false;
        return;
      }
      var_scope = return_type.is_void()
        ? scope(void_function_argument_origin)
        : scope(function_argument_origin);
      pass = true;
    }
    boost::phoenix::function<set_void_function> set_void_function_f;

    void set_allows_sampling_origin::operator()(const std::string& identifier,
                                                scope& var_scope) const {
      if (ends_with("_lp", identifier)) {
        var_scope = var_scope.void_fun()
          ? scope(void_function_argument_origin_lp)
          : scope(function_argument_origin_lp);
      } else if (ends_with("_rng", identifier)) {
        var_scope = var_scope.void_fun()
          ? scope(void_function_argument_origin_rng)
          : scope(function_argument_origin_rng);
      } else {
        var_scope = var_scope.void_fun()
          ? scope(void_function_argument_origin)
          : scope(function_argument_origin);
      }
    }
    boost::phoenix::function<set_allows_sampling_origin>
    set_allows_sampling_origin_f;

    void validate_declarations::operator()(bool& pass,
                                           std::set<std::pair<std::string,
                                           function_signature_t> >& declared,
                                           std::set<std::pair<std::string,
                                           function_signature_t> >& defined,
                                           std::ostream& error_msgs,
                                           bool allow_undefined) const {
      using std::set;
      using std::string;
      using std::pair;
      typedef set<pair<string, function_signature_t> >::iterator iterator_t;
      if (!allow_undefined) {
        for (iterator_t it = declared.begin(); it != declared.end(); ++it) {
          if (defined.find(*it) == defined.end()) {
            error_msgs <<"Function declared, but not defined."
                       << " Function name=" << (*it).first
                       << std::endl;
            pass = false;
            return;
          }
        }
      }
      pass = true;
    }
    boost::phoenix::function<validate_declarations> validate_declarations_f;


    bool fun_exists(const std::set<std::pair<std::string,
                    function_signature_t> >& existing,
                    const std::pair<std::string,
                    function_signature_t>& name_sig,
                    bool name_only = true) {
      for (std::set<std::pair<std::string,
             function_signature_t> >::const_iterator it
             = existing.begin();
           it != existing.end();
           ++it) {
        if (name_sig.first == (*it).first
            && (name_only
                || name_sig.second.second == (*it).second.second))
          return true;  // name and arg sequences match
      }
      return false;
    }

    void validate_prob_fun::operator()(std::string& fname, bool& pass,
                                       std::ostream& error_msgs) const {
      if (has_prob_fun_suffix(fname)) {
        std::string dist_name = strip_prob_fun_suffix(fname);
        if (!fun_name_exists(fname)  // catch redefines later avoid fwd
            && (fun_name_exists(dist_name + "_lpdf")
                || fun_name_exists(dist_name + "_lpmf")
                || fun_name_exists(dist_name + "_log"))) {
          error_msgs << "Parse Error.  Probability function already defined"
                     << " for " << dist_name << std::endl;
          pass = false;
          return;
        }
      }
      if (has_cdf_suffix(fname)) {
        std::string dist_name = strip_cdf_suffix(fname);
        if (fun_name_exists(dist_name + "_cdf_log")
            || fun_name_exists(dist_name + "_lcdf")) {
          error_msgs << " Parse Error.  CDF already defined for "
                     << dist_name << std::endl;
          pass = false;
          return;
        }
      }
      if (has_ccdf_suffix(fname)) {
        std::string dist_name = strip_ccdf_suffix(fname);
        if (fun_name_exists(dist_name + "_ccdf_log")
            || fun_name_exists(dist_name + "_lccdf")) {
          error_msgs << " Parse Error.  CCDF already defined for "
                     << dist_name << std::endl;
          pass = false;
          return;
        }
      }
    }
    boost::phoenix::function<validate_prob_fun> validate_prob_fun_f;

    void add_function_signature::operator()(const function_decl_def& decl,
        bool& pass,
        std::set<std::pair<std::string, function_signature_t> >&
                                            functions_declared,
        std::set<std::pair<std::string, function_signature_t> >&
                                            functions_defined,
        std::ostream& error_msgs) const {
      // build up representations
      expr_type result_type(decl.return_type_.base_type_,
                            decl.return_type_.num_dims_);
      std::vector<function_arg_type> arg_types;
      for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
        arg_types.push_back(
            function_arg_type(expr_type(decl.arg_decls_[i].arg_type_.base_type_,
                                        decl.arg_decls_[i].arg_type_.num_dims_),
                              decl.arg_decls_[i].is_data_));

      function_signature_t sig(result_type, arg_types);
      std::pair<std::string, function_signature_t> name_sig(decl.name_, sig);

      // check that not already declared if just declaration
      if (decl.body_.is_no_op_statement()
          && fun_exists(functions_declared, name_sig)) {
        error_msgs << "Parse Error.  Function already declared, name="
                   << decl.name_;
        pass = false;
        return;
      }

      // check not already user defined
      if (fun_exists(functions_defined, name_sig)) {
        error_msgs << "Parse Error.  Function already defined, name="
                   << decl.name_;
        pass = false;
        return;
      }

      // check not already system defined
      if (!fun_exists(functions_declared, name_sig)
          && function_signatures::instance().is_defined(decl.name_, sig)) {
        error_msgs << "Parse Error.  Function system defined, name="
                   << decl.name_;
        pass = false;
        return;
      }

       // check argument qualifiers
      if (!decl.body_.is_no_op_statement()) {
        function_signature_t decl_sig =
          function_signatures::instance().get_definition(decl.name_, sig);
        if (!decl_sig.first.is_ill_formed()) {
          if (decl_sig.second.size() != arg_types.size()) {
            error_msgs << "Declaration, definition mismatch for function "
                       << decl.name_
                       << ": declaration has " << arg_types.size()
                       << " arguments, definition has " << arg_types.size()
                       << "arguments." << std::endl;
            pass = false;
            return;
          }
          for (size_t i = 0; i < arg_types.size(); ++i) {
            if (decl_sig.second[i].expr_type_ != arg_types[i].expr_type_
                || decl_sig.second[i].data_only_ != arg_types[i].data_only_) {
              error_msgs << "Declaration, definition mismatch for function "
                         << decl.name_ << " argument " << (i + 1)
                         << ": argument declared as "
                         << (decl_sig.second[i].data_only_ ? "" : "data ")
                         << decl_sig.second[i].expr_type_
                         << ", defined as "
                         << (arg_types[i].data_only_ ? "" : "data ")
                         << arg_types[i].expr_type_ << "." <<  std::endl;
              pass = false;
              return;
            }
          }
        }
      }

      if (ends_with("_lpdf", decl.name_)
          && arg_types[0].expr_type_.base_type_ == INT_T) {
        error_msgs << "Parse Error.  Probability density functions require"
                   << " real variates (first argument)."
                   << " Found type = " << arg_types[0].expr_type_ << std::endl;
        pass = false;
        return;
      }
      if (ends_with("_lpmf", decl.name_)
          && arg_types[0].expr_type_.base_type_ != INT_T) {
        error_msgs << "Parse Error.  Probability mass functions require"
                   << " integer variates (first argument)."
                   << " Found type = " << arg_types[0].expr_type_ << std::endl;
        pass = false;
        return;
      }

      // add declaration in local sets and in parser function sigs
      if (functions_declared.find(name_sig) == functions_declared.end()) {
        functions_declared.insert(name_sig);
        function_signatures::instance()
          .add(decl.name_, result_type, arg_types);
        function_signatures::instance().set_user_defined(name_sig);
      }

      // add as definition if there's a body
      if (!decl.body_.is_no_op_statement())
        functions_defined.insert(name_sig);
      pass = true;
    }
    boost::phoenix::function<add_function_signature> add_function_signature_f;

    void validate_pmf_pdf_variate::operator()(function_decl_def& decl,
                                              bool& pass,
                                              std::ostream& error_msgs)
      const {
      if (!has_prob_fun_suffix(decl.name_))
        return;
      if (decl.arg_decls_.size() == 0) {
        error_msgs << "Parse Error.  Probability functions require"
                   << " at least one argument." << std::endl;
        pass = false;
        return;
      }
      expr_type variate_type = decl.arg_decls_[0].arg_type_;
      if (ends_with("_lpdf", decl.name_) && variate_type.base_type_ == INT_T) {
        error_msgs << "Parse Error.  Probability density functions require"
                   << " real variates (first argument)."
                   << " Found type = " << variate_type << std::endl;
        pass = false;
        return;
      }
      if (ends_with("_lpmf", decl.name_) && variate_type.base_type_ != INT_T) {
        error_msgs << "Parse Error.  Probability mass functions require"
                   << " integer variates (first argument)."
                   << " Found type = " << variate_type << std::endl;
        pass = false;
        return;
      }
    }
    boost::phoenix::function<validate_pmf_pdf_variate>
    validate_pmf_pdf_variate_f;

    void validate_return_type::operator()(function_decl_def& decl,
                                          bool& pass,
                                          std::ostream& error_msgs) const {
      pass = decl.body_.is_no_op_statement()
        || stan::lang::returns_type(decl.return_type_, decl.body_,
                                    error_msgs);
      if (!pass) {
        error_msgs << "Improper return in body of function." << std::endl;
        return;
      }

      if ((ends_with("_log", decl.name_)
           || ends_with("_lpdf", decl.name_)
           || ends_with("_lpmf", decl.name_)
           || ends_with("_lcdf", decl.name_)
           || ends_with("_lccdf", decl.name_))
          && !decl.return_type_.is_primitive_double()) {
        pass = false;
        error_msgs << "Require real return type for probability functions"
                   << " ending in _log, _lpdf, _lpmf, _lcdf, or _lccdf."
                   << std::endl;
      }
    }
    boost::phoenix::function<validate_return_type> validate_return_type_f;

    void set_fun_params_scope::operator()(scope& var_scope, variable_map& vm)
      const {
      var_scope = scope(var_scope.program_block(), true);
      // TODO(morris): remove if params_r__ no longer used
      vm.add("params_r__", VECTOR_T, parameter_origin);
    }
    boost::phoenix::function<set_fun_params_scope> set_fun_params_scope_f;

    void unscope_variables::operator()(function_decl_def& decl,
                                       variable_map& vm) const {
      vm.remove("params_r__");
      for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
        vm.remove(decl.arg_decls_[i].name_);
    }
    boost::phoenix::function<unscope_variables> unscope_variables_f;

    void add_fun_var::operator()(arg_decl& decl, scope& scope, bool& pass,
                                 variable_map& vm,
                                 std::ostream& error_msgs) const {
      if (vm.exists(decl.name_)) {
        pass = false;
        error_msgs << "duplicate declaration of variable, name="
                   << decl.name_
                   << "; attempt to redeclare as function argument"
                   << "; original declaration as ";
        print_scope(error_msgs, vm.get_scope(decl.name_));
        error_msgs << std::endl;
        return;
      }
      pass = true;
      origin_block var_origin = scope.program_block();
      if (var_origin == data_origin) {
        if (decl.base_variable_declaration().base_type_ == INT_T) {
          pass = false;
          error_msgs << "data qualifier cannot be applied to int variable, "
                     << "name " << decl.name_;
          error_msgs << std::endl;
          return;
        }
        decl.is_data_ = true;
      } else {
        var_origin = function_argument_origin;
      }
      vm.add(decl.name_, decl.base_variable_declaration(),
             var_origin);
    }
    boost::phoenix::function<add_fun_var> add_fun_var_f;

    // TODO(carpenter): seems redundant; see if it can be removed
    void set_omni_idx::operator()(omni_idx& val) const {
      val = omni_idx();
    }
    boost::phoenix::function<set_omni_idx> set_omni_idx_f;

    void validate_int_expr_silent::operator()(const expression & e, bool& pass)
      const {
      pass = e.expression_type().is_primitive_int();
    }
    boost::phoenix::function<validate_int_expr_silent>
    validate_int_expr_silent_f;

    void validate_int_expression_warn::operator()(const expression & e,
                                                  bool& pass,
                                                  std::ostream& error_msgs)
      const {
      if (e.expression_type().type() != INT_T) {
        error_msgs << "ERROR:  Indexes must be expressions of integer type."
                   << " found type = ";
        write_base_expr_type(error_msgs, e.expression_type().type());
        error_msgs << '.' << std::endl;
      }
      pass = e.expression_type().is_primitive_int();
    }
    boost::phoenix::function<validate_int_expression_warn>
    validate_int_expression_warn_f;


    void validate_ints_expression::operator()(const expression& e, bool& pass,
                                              std::ostream& error_msgs) const {
      if (e.expression_type().type() != INT_T) {
        error_msgs << "ERROR:  Container index must be integer; found type=";
        write_base_expr_type(error_msgs, e.expression_type().type());
        error_msgs << std::endl;
        pass = false;
        return;
      }
      if (e.expression_type().num_dims_ > 1) {
        // tests > 1 so that message is coherent because the single
        // integer array tests don't print
        error_msgs << "index must be integer or 1D integer array;"
                   << " found number of dimensions="
                   << e.expression_type().num_dims_
                   << std::endl;
        pass = false;
        return;
      }
      if (e.expression_type().num_dims_ == 0) {
        // need integer array expression here, but nothing else to report
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_ints_expression>
    validate_ints_expression_f;


    void add_params_var::operator()(variable_map& vm) const {
      vm.add("params_r__",
             base_var_decl("params_r__", std::vector<expression>(), VECTOR_T),
             parameter_origin);  // acts like a parameter
    }
    boost::phoenix::function<add_params_var> add_params_var_f;

    void remove_params_var::operator()(variable_map& vm) const {
      vm.remove("params_r__");
    }
    boost::phoenix::function<remove_params_var> remove_params_var_f;

    void dump_program_line(size_t idx_errline, int offset,
                           const std::string& origin_file,
                           size_t origin_line,
                           const io::program_reader& reader,
                           const std::vector<std::string>& program_lines,
                           std::stringstream& error_msgs) {
      boost::format fmt_lineno("%6d: ");
      if (idx_errline + offset > 0
          && idx_errline + offset < program_lines.size()) {
        io::program_reader::trace_t trace = reader.trace(idx_errline + offset);
        if (trace[trace.size() - 1].first == origin_file) {
          std::string lineno = str(fmt_lineno % (origin_line + offset));
          error_msgs << lineno << program_lines[idx_errline + offset - 1]
                     << std::endl;
        }
      }
    }

    void program_error::operator()(pos_iterator_t begin, pos_iterator_t end,
                                   pos_iterator_t where, variable_map& vm,
                                   std::stringstream& error_msgs,
                                   const io::program_reader& reader) const {
      // extract line and column of error
      size_t idx_errline = boost::spirit::get_line(where);
      if (idx_errline == 0) {
        error_msgs << "Error before start of program." << std::endl;
        return;
      }
      size_t idx_errcol = 0;
      idx_errcol = get_column(begin, where) - 1;

      // extract lines of included program
      std::basic_stringstream<char> program_ss;
      program_ss << boost::make_iterator_range(begin, end);
      std::vector<std::string> program_lines;
      while (!program_ss.eof()) {
        std::string line;
        std::getline(program_ss, line);
        program_lines.push_back(line);
      }

      // dump include trace for error line
      io::program_reader::trace_t trace = reader.trace(idx_errline);
      std::string origin_file = trace[trace.size() - 1].first;
      size_t origin_line = trace[trace.size() - 1].second;
      error_msgs << "  error in '" << trace[trace.size() - 1].first
                 << "' at line " << trace[trace.size() - 1].second
                 << ", column " << idx_errcol << std::endl;
      for (int i = trace.size() - 1; i-- > 0; )
        error_msgs << "  included from '" << trace[i].first
                   << "' at line " << trace[i].second << std::endl;

      // dump context of error
      error_msgs << "  -------------------------------------------------"
                 << std::endl;

      dump_program_line(idx_errline, -2, origin_file, origin_line,
                        reader, program_lines, error_msgs);
      dump_program_line(idx_errline, -1, origin_file, origin_line,
                        reader, program_lines, error_msgs);
      dump_program_line(idx_errline, 0, origin_file, origin_line,
                        reader, program_lines, error_msgs);
      error_msgs << std::setw(idx_errcol + 8) << "^" << std::endl;
      dump_program_line(idx_errline, +1, origin_file, origin_line,
                        reader, program_lines, error_msgs);

      error_msgs << "  -------------------------------------------------"
                 << std::endl << std::endl;
    }
    boost::phoenix::function<program_error> program_error_f;

    void add_conditional_condition::operator()(conditional_statement& cs,
                                               const expression& e,
                                               bool& pass,
                                               std::stringstream& error_msgs)
      const {
      if (!e.expression_type().is_primitive()) {
        error_msgs << "conditions in if-else statement must be"
                   << " primitive int or real;"
                   << " found type=" << e.expression_type()
                   << std::endl;
        pass = false;
        return;
      }
      cs.conditions_.push_back(e);
      pass = true;
      return;
    }
    boost::phoenix::function<add_conditional_condition>
    add_conditional_condition_f;

    void add_conditional_body::operator()(conditional_statement& cs,
                                          const statement& s) const {
      cs.bodies_.push_back(s);
    }
    boost::phoenix::function<add_conditional_body> add_conditional_body_f;

    void deprecate_old_assignment_op::operator()(std::ostream& error_msgs)
      const {
      error_msgs << "Warning (non-fatal): assignment operator <- deprecated"
                 << " in the Stan language;"
                 << " use = instead."
                 << std::endl;
    }
    boost::phoenix::function<deprecate_old_assignment_op>
    deprecate_old_assignment_op_f;

    void validate_return_allowed::operator()(scope var_scope, bool& pass,
                                             std::ostream& error_msgs) const {
      if (!var_scope.non_void_fun()) {
        error_msgs << "Returns only allowed from function bodies."
                   << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_return_allowed> validate_return_allowed_f;

    void validate_void_return_allowed::operator()(scope var_scope,
                                                  bool& pass,
                                                  std::ostream& error_msgs)
      const {
      if (!var_scope.void_fun()) {
        error_msgs << "Void returns only allowed from function"
                   << " bodies of void return type."
                   << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_void_return_allowed>
    validate_void_return_allowed_f;

    void validate_lhs_var_assgn::operator()(const std::string& name,
                                       const scope& var_scope,
                                       variable& v,  bool& pass,
                                       const variable_map& vm,
                                       std::ostream& error_msgs) const {
      // validate existence
      if (!vm.exists(name)) {
        // fail silently, allow backtracking
        pass = false;
        return;
      }
      // validate scope matches declaration scope
      scope lhs_origin = vm.get_scope(name);
      if (lhs_origin.program_block() != var_scope.program_block()) {
        pass = false;
        return;
      }
      v = variable(name);
      v.set_type(vm.get_base_type(name), vm.get_num_dims(name));
      pass = true;
    }
    boost::phoenix::function<validate_lhs_var_assgn> validate_lhs_var_assgn_f;

    void validate_assgn::operator()(const assgn& a, bool& pass,
                                    const variable_map& vm,
                                    std::ostream& error_msgs) const {
      // validate var exists
      std::string name = a.lhs_var_.name_;
      if (!vm.exists(name)) {
        error_msgs << "Unknown variable in assignment"
                   << "; lhs variable=" << name
                   << std::endl;
        pass = false;
        return;
      }

      expression lhs_expr = expression(a.lhs_var_);
      expr_type lhs_type = indexed_type(lhs_expr, a.idxs_);
      if (lhs_type.is_ill_formed()) {
        error_msgs << "Left-hand side indexing incompatible with variable."
                   << std::endl;
        pass = false;
        return;
      }
      if (!has_same_shape(lhs_type, a.rhs_, name, "assignment", error_msgs)) {
        pass = false;
        return;
      }
      if (a.lhs_var_occurs_on_rhs()) {
        // this only requires a warning --- a deep copy will be made
        error_msgs << "WARNING: left-hand side variable"
                   << " (name=" << name << ")"
                   << " occurs on right-hand side of assignment, causing"
                   << " inefficient deep copy to avoid aliasing."
                   << std::endl;
      }

      pass = true;
    }
    boost::phoenix::function<validate_assgn> validate_assgn_f;

    void validate_assignment::operator()(assignment& a,
                                         const scope& var_scope,
                                         bool& pass, const variable_map& vm,
                                         std::ostream& error_msgs) const {
      std::string name = a.var_dims_.name_;
      if (!vm.exists(name)) {
        error_msgs << "Unknown variable in assignment"
                   << "; lhs variable=" << name
                   << std::endl;
        pass = false;
        return;
      }
      if (!can_assign_to_lhs_var(name, var_scope, vm, error_msgs)) {
        pass = false;
        return;
      }
      a.var_type_ = vm.get(name);
      expr_type inferred_lhs_type
        = infer_var_dims_type(a.var_type_, a.var_dims_);
      if (inferred_lhs_type.is_ill_formed()) {
        error_msgs << "Too many indexes for variable"
                   << "; variable name = "
                   << name
                   << "; num dimensions given = "
                   << a.var_dims_.dims_.size()
                   << "; variable array dimensions = "
                   << a.var_type_.dims_.size()
                   << std::endl;
        pass = false;
        return;
      }
      if (!has_same_shape(inferred_lhs_type, a.expr_, name,
                          "assignment", error_msgs)) {
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_assignment> validate_assignment_f;

    void validate_compound_assignment::operator()(compound_assignment& ca,
                                         const scope& var_scope,
                                         bool& pass, variable_map& vm,
                                         std::ostream& error_msgs) const {
      std::string name = ca.var_dims_.name_;
      if (!vm.exists(name)) {
        error_msgs << "Unknown variable in compound assignment"
                   << "; lhs variable=" << name
                   << std::endl;
        pass = false;
        return;
      }
      if (!can_assign_to_lhs_var(name, var_scope, vm, error_msgs)) {
        pass = false;
        return;
      }
      ca.var_type_ = vm.get(name);
      expr_type inferred_lhs_type
        = infer_var_dims_type(ca.var_type_, ca.var_dims_);
      int lhs_num_dims = ca.var_type_.dims_.size();
      int lhs_num_idxs = ca.var_dims_.dims_.size();

      if (inferred_lhs_type.is_ill_formed()) {
        error_msgs << "Too many indexes for variable"
                   << "; variable name = " << name
                   << "; specified indexes = " << lhs_num_idxs
                   << "; variable array dimensions = " << lhs_num_dims
                   << std::endl;
        pass = false;
        return;
      }
      // no compound assign for array types  (std::vector)
      std::string op_equals = ca.op_;
      ca.op_ = op_equals.substr(0, op_equals.size()-1);
      if (inferred_lhs_type.num_dims() > 0) {
        error_msgs << "Cannot apply operator '" << op_equals
                   << "' to array variable; variable name = "
                   << name
                   << ".";
        error_msgs << std::endl;
        pass = false;
        return;
      }

      expr_type lhs_type = inferred_lhs_type.type();
      expr_type rhs_type = ca.expr_.expression_type();
      if (lhs_type.is_primitive()
          && boost::algorithm::starts_with(ca.op_, ".")) {
        error_msgs << "Cannot apply element-wise operation to scalar"
                   << "; compound operator is: " << op_equals
                   << std::endl;
        pass = false;
        return;
      }
      if (lhs_type.is_primitive()
          && rhs_type.is_primitive()
          && (lhs_type == DOUBLE_T || lhs_type == rhs_type)) {
        // done checking <prim> <op>= <prim>
        pass = true;
        return;
      }

      bool types_compatible =
        // container types allow infix and element-wise operations
        // when lhs and rhs are same shape, and broadcast operations
        // when rhs is double and lhs is vector, row_vector, or matrix
        (lhs_type == rhs_type
         || (lhs_type == VECTOR_T && rhs_type == DOUBLE_T)
         || (lhs_type == ROW_VECTOR_T && rhs_type == DOUBLE_T)
         || (lhs_type == ROW_VECTOR_T && rhs_type == MATRIX_T)
         || (lhs_type == MATRIX_T && rhs_type == DOUBLE_T));
      if (!types_compatible) {
        error_msgs << "Cannot apply operator '" << op_equals
                   << "' to operands;"
                   << " left-hand side type = " << lhs_type
                   << "; right-hand side type=" << rhs_type
                   << std::endl;
        pass = false;
        return;
      }
      std::string op_name;
      if (ca.op_ == "+") {
        op_name = "add";
      } else if (ca.op_ == "-") {
        op_name = "subtract";
      } else if (ca.op_ == "*") {
        op_name = "multiply";
      } else if (ca.op_ == "/") {
        op_name = "divide";
      } else if (ca.op_ == "./") {
        op_name = "elt_divide";
      } else if (ca.op_ == ".*") {
        op_name = "elt_multiply";
      }
      // check that "lhs <op> rhs" is valid stan::math function sig
      std::vector<function_arg_type> arg_types;
      arg_types.push_back(function_arg_type(lhs_type));
      arg_types.push_back(function_arg_type(rhs_type));
      function_signature_t op_equals_sig(lhs_type, arg_types);
      if (!function_signatures::instance().is_defined(op_name, op_equals_sig)) {
        error_msgs << "Cannot apply operator '" << op_equals
                   << "' to operands;"
                   << " left-hand side type = " << lhs_type
                   << "; right-hand side type=" << rhs_type
                   << std::endl;
        pass = false;
        return;
      }
      ca.op_name_ = op_name;
      pass = true;
    }
    boost::phoenix::function<validate_compound_assignment>
    validate_compound_assignment_f;

    void validate_sample::operator()(sample& s,
                                     const variable_map& var_map, bool& pass,
                                     std::ostream& error_msgs) const {
      static const bool user_facing = true;
      std::vector<expr_type> arg_types;
      arg_types.push_back(s.expr_.expression_type());
      for (size_t i = 0; i < s.dist_.args_.size(); ++i)
        arg_types.push_back(s.dist_.args_[i].expression_type());
      std::string function_name(s.dist_.family_);
      std::string internal_function_name = get_prob_fun(function_name);
      s.is_discrete_ = function_signatures::instance()
        .discrete_first_arg(internal_function_name);

      if (internal_function_name.size() == 0) {
        pass = false;
        error_msgs << "Error: couldn't find distribution named "
                   << function_name << std::endl;
        return;
      }

      if (!(ends_with("_lpdf", internal_function_name)
            || ends_with("_lpmf", internal_function_name)
            || ends_with("_log", internal_function_name))) {
        pass = false;
        error_msgs << "Probability function must end in _lpdf or _lpmf."
                   << " Found distribution family = " << function_name
                   << " with no corresponding probability function"
                   << " " << function_name << "_lpdf"
                   << ", " << function_name << "_lpmf"
                   << ", or " << function_name << "_log" << std::endl;
        return;
      }

      if ((internal_function_name.find("multiply_log") != std::string::npos)
          || (internal_function_name.find("binomial_coefficient_log")
              != std::string::npos)) {
        error_msgs << "Only distribution names can be used with"
                   << " sampling (~) notation; found non-distribution"
                   << " function: " << function_name
                   << std::endl;
        pass = false;
        return;
      }

      if (internal_function_name.find("cdf_log") != std::string::npos) {
        error_msgs << "CDF and CCDF functions may not be used with"
                   << " sampling notation."
                   << " Use increment_log_prob("
                   << internal_function_name << "(...)) instead."
                   << std::endl;
        pass = false;
        return;
      }

      if (internal_function_name == "lkj_cov_log") {
        error_msgs << "Warning: the lkj_cov_log() sampling distribution"
                   << " is deprecated.  It will be removed in Stan 3."
                   << std::endl
                   << "Code LKJ covariance in terms of an lkj_corr()"
                   << " distribution on a correlation matrix"
                   << " and independent lognormals on the scales."
                   << std::endl << std::endl;
      }

      if (!is_double_return(internal_function_name, arg_types, error_msgs)) {
        error_msgs << "require real scalar return type for"
                   << " probability function." << std::endl;
        pass = false;
        return;
      }
      // test for LHS not being purely a variable
      if (has_non_param_var(s.expr_, var_map)) {
        error_msgs << "Warning (non-fatal):"
                   << std::endl
                   << "Left-hand side of sampling statement (~) may contain a"
                   << " non-linear transform of a parameter or local variable."
                   << std::endl
                   << "If it does, you need to include a target += statement"
                   << " with the log absolute determinant of the Jacobian of"
                   << " the transform."
                   << std::endl
                   << "Left-hand-side of sampling statement:"
                   << std::endl
                   << "    ";
        generate_expression(s.expr_, user_facing, error_msgs);
        error_msgs << " ~ " << function_name << "(...)"
                   << std::endl;
      }
      // validate that variable and params are univariate if truncated
      if (s.truncation_.has_low() || s.truncation_.has_high()) {
        if (!is_univariate(s.expr_.expression_type())) {
          error_msgs << "Outcomes in truncated distributions"
                     << " must be univariate."
                     << std::endl
                     << "  Found outcome expression: ";
          generate_expression(s.expr_, user_facing, error_msgs);
          error_msgs << std::endl
                     << "  with non-univariate type: "
                     << s.expr_.expression_type()
                     << std::endl;
          pass = false;
          return;
        }
        for (size_t i = 0; i < s.dist_.args_.size(); ++i)
          if (!is_univariate(s.dist_.args_[i].expression_type())) {
            error_msgs << "Parameters in truncated distributions"
                       << " must be univariate."
                       << std::endl
                       << "  Found parameter expression: ";
            generate_expression(s.dist_.args_[i], user_facing, error_msgs);
            error_msgs << std::endl
                       << "  with non-univariate type: "
                       << s.dist_.args_[i].expression_type()
                       << std::endl;
            pass = false;
            return;
          }
      }
      if (s.truncation_.has_low()
          && !is_univariate(s.truncation_.low_.expression_type())) {
        error_msgs << "Lower bounds in truncated distributions"
                   << " must be univariate."
                   << std::endl
                   << "  Found lower bound expression: ";
        generate_expression(s.truncation_.low_, user_facing, error_msgs);
        error_msgs << std::endl
                   << "  with non-univariate type: "
                   << s.truncation_.low_.expression_type()
                   << std::endl;
        pass = false;
        return;
      }
      if (s.truncation_.has_high()
          && !is_univariate(s.truncation_.high_.expression_type())) {
        error_msgs << "Upper bounds in truncated distributions"
                   << " must be univariate."
                   << std::endl
                   << "  Found upper bound expression: ";
        generate_expression(s.truncation_.high_, user_facing, error_msgs);
        error_msgs << std::endl
                   << "  with non-univariate type: "
                   << s.truncation_.high_.expression_type()
                   << std::endl;
        pass = false;
        return;
      }

      // make sure CDFs or CCDFs exist with conforming signature
      // T[L, ]
      if (s.truncation_.has_low() && !s.truncation_.has_high()) {
        std::vector<expr_type> arg_types_trunc(arg_types);
        arg_types_trunc[0] = s.truncation_.low_.expression_type();
        std::string function_name_ccdf = get_ccdf(s.dist_.family_);
        if (function_name_ccdf == s.dist_.family_
            || !is_double_return(function_name_ccdf, arg_types_trunc,
                              error_msgs)) {
          error_msgs << "lower truncation not defined for specified"
                     << " arguments to "
                     << s.dist_.family_ << std::endl;
          pass = false;
          return;
        }
        if (!is_double_return(function_name_ccdf, arg_types, error_msgs)) {
          error_msgs << "lower bound in truncation type does not match"
                     << " sampled variate in distribution's type"
                     << std::endl;
          pass = false;
          return;
        }
      }
      // T[, H]
      if (!s.truncation_.has_low() && s.truncation_.has_high()) {
        std::vector<expr_type> arg_types_trunc(arg_types);
        arg_types_trunc[0] = s.truncation_.high_.expression_type();
        std::string function_name_cdf = get_cdf(s.dist_.family_);
        if (function_name_cdf == s.dist_.family_
            || !is_double_return(function_name_cdf, arg_types_trunc,
                                 error_msgs)) {
          error_msgs << "upper truncation not defined for"
                     << " specified arguments to "
                     << s.dist_.family_ << std::endl;

          pass = false;
          return;
        }
        if (!is_double_return(function_name_cdf, arg_types, error_msgs)) {
          error_msgs << "upper bound in truncation type does not match"
                     << " sampled variate in distribution's type"
                     << std::endl;
          pass = false;
          return;
        }
      }
      // T[L, H]
      if (s.truncation_.has_low() && s.truncation_.has_high()) {
        std::vector<expr_type> arg_types_trunc(arg_types);
        arg_types_trunc[0] = s.truncation_.low_.expression_type();
        std::string function_name_cdf = get_cdf(s.dist_.family_);
        if (function_name_cdf == s.dist_.family_
            || !is_double_return(function_name_cdf, arg_types_trunc,
                                 error_msgs)) {
          error_msgs << "lower truncation not defined for specified"
                     << " arguments to "
                     << s.dist_.family_ << std::endl;
          pass = false;
          return;
        }
        if (!is_double_return(function_name_cdf, arg_types, error_msgs)) {
          error_msgs << "lower bound in truncation type does not match"
                     << " sampled variate in distribution's type"
                     << std::endl;
          pass = false;
          return;
        }
      }
      pass = true;
    }
    boost::phoenix::function<validate_sample> validate_sample_f;

    void expression_as_statement::operator()(bool& pass,
                                     const stan::lang::expression& expr,
                                     std::stringstream& error_msgs) const {
      static const bool user_facing = true;
      if (expr.expression_type() != VOID_T) {
        error_msgs << "Illegal statement beginning with non-void"
                   << " expression parsed as"
                   << std::endl << "  ";
        generate_expression(expr.expr_, user_facing, error_msgs);
        error_msgs << std::endl
                   << "Not a legal assignment, sampling, or function"
                   << " statement.  Note that"
                   << std::endl
                   << "  * Assignment statements only allow variables"
                   << " (with optional indexes) on the left;"
                   << std::endl
                   << "    if you see an outer function logical_lt (<)"
                   << " with negated (-) second argument,"
                   << std::endl
                   << "    it indicates an assignment statement A <- B"
                   << " with illegal left"
                   << std::endl
                   << "    side A parsed as expression (A < (-B))."
                   << std::endl
                   << "  * Sampling statements allow arbitrary"
                   << " value-denoting expressions on the left."
                   << std::endl
                   << "  * Functions used as statements must be"
                   << " declared to have void returns"
                   << std::endl << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<expression_as_statement> expression_as_statement_f;

    void unscope_locals::operator()(const std::vector<var_decl>& var_decls,
                                    variable_map& vm) const {
      for (size_t i = 0; i < var_decls.size(); ++i)
        vm.remove(var_decls[i].name());
    }
    boost::phoenix::function<unscope_locals> unscope_locals_f;

    void add_while_condition::operator()(while_statement& ws,
                                         const expression& e,  bool& pass,
                                         std::stringstream& error_msgs) const {
      pass = e.expression_type().is_primitive();
      if (!pass) {
        error_msgs << "conditions in while statement must be primitive"
                   << " int or real;"
                   << " found type=" << e.expression_type() << std::endl;
        return;
      }
      ws.condition_ = e;
    }
    boost::phoenix::function<add_while_condition> add_while_condition_f;

    void add_while_body::operator()(while_statement& ws, const statement& s)
      const {
      ws.body_ = s;
    }
    boost::phoenix::function<add_while_body> add_while_body_f;

    void add_loop_identifier::operator()(const std::string& name,
                                         std::string& name_local,
                                         const scope& var_scope,
                                         bool& pass, variable_map& vm,
                                         std::stringstream& error_msgs) const {
      name_local = name;
      pass = !vm.exists(name);
      if (!pass)
        error_msgs << "ERROR: loop variable already declared."
                   << " variable name=\"" << name << "\"" << std::endl;
      else
        vm.add(name, base_var_decl(name, std::vector<expression>(), INT_T),
               scope(var_scope.program_block(), true));
    }
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    void remove_loop_identifier::operator()(const std::string& name,
                                            variable_map& vm) const {
      vm.remove(name);
    }
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;

    void validate_int_expr::operator()(const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      if (!expr.expression_type().is_primitive_int()) {
        error_msgs << "expression denoting integer required; found type="
                   << expr.expression_type() << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_int_expr> validate_int_expr_f;

    void deprecate_increment_log_prob::operator()(
                                       std::stringstream& error_msgs) const {
      error_msgs << "Warning (non-fatal): increment_log_prob(...);"
                 << " is deprecated and will be removed in the future."
                 << std::endl
                 << "  Use target += ...; instead."
                 << std::endl;
    }
    boost::phoenix::function<deprecate_increment_log_prob>
    deprecate_increment_log_prob_f;

    void validate_allow_sample::operator()(const scope& var_scope,
                                           bool& pass,
                                           std::stringstream& error_msgs)
      const {
      pass = var_scope.allows_sampling();
      if (!pass)
        error_msgs << "Sampling statements (~) and increment_log_prob() are"
                   << std::endl
                   << "only allowed in the model block or lp functions."
                   << std::endl;
    }
    boost::phoenix::function<validate_allow_sample> validate_allow_sample_f;

    void validate_non_void_expression::operator()(const expression& e,
                                                  bool& pass,
                                                  std::ostream& error_msgs)
      const {
      pass = !e.expression_type().is_void();
      if (!pass)
        error_msgs << "attempt to increment log prob with void expression"
                   << std::endl;
    }
    boost::phoenix::function<validate_non_void_expression>
    validate_non_void_expression_f;

    template <typename T, typename I>
    void add_line_number::operator()(T& line,
                                     const I& begin,
                                     const I& end) const {
      line.begin_line_ = get_line(begin);
      line.end_line_ = get_line(end);
    }
    boost::phoenix::function<add_line_number>
    add_line_number_f;

    template void add_line_number::operator()(var_decl&,
                                              const pos_iterator_t& begin,
                                              const pos_iterator_t& end) const;


    template void add_line_number::operator()(statement&,
                                              const pos_iterator_t& begin,
                                              const pos_iterator_t& end) const;


    void set_void_return::operator()(return_statement& s) const {
      s = return_statement();
    }
    boost::phoenix::function<set_void_return> set_void_return_f;

    void set_no_op::operator()(no_op_statement& s) const {
      s = no_op_statement();
    }
    boost::phoenix::function<set_no_op> set_no_op_f;


    void deprecated_integrate_ode::operator()(std::ostream& error_msgs)
      const {
      error_msgs << "Warning: the integrate_ode() function is deprecated"
             << " in the Stan language; use integrate_ode_rk45() [non-stiff]"
             << " or integrate_ode_bdf() [stiff] instead."
             << std::endl;
    }
    boost::phoenix::function<deprecated_integrate_ode>
    deprecated_integrate_ode_f;

    template <class T>
    void validate_integrate_ode_non_control_args(const T& ode_fun,
                                                 const variable_map& var_map,
                                                 bool& pass,
                                                 std::ostream& error_msgs) {
      pass = true;
      // test function argument type
      expr_type sys_result_type(DOUBLE_T, 1);
      std::vector<function_arg_type> sys_arg_types;
      sys_arg_types.push_back(function_arg_type(expr_type(DOUBLE_T, 0)));
      sys_arg_types.push_back(function_arg_type(expr_type(DOUBLE_T, 1)));
      sys_arg_types.push_back(function_arg_type(expr_type(DOUBLE_T, 1)));
      sys_arg_types.push_back(function_arg_type(expr_type(DOUBLE_T, 1)));
      sys_arg_types.push_back(function_arg_type(expr_type(INT_T, 1)));
      function_signature_t system_signature(sys_result_type, sys_arg_types);
      if (!function_signatures::instance()
          .is_defined(ode_fun.system_function_name_, system_signature)) {
        error_msgs << "first argument to "
                   << ode_fun.integration_function_name_
                   << " must be the name of a function with signature"
                   << " (real, real[], real[], real[], int[]) : real[] ";
        pass = false;
      }

      // test regular argument types
      if (ode_fun.y0_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "second argument to "
                   << ode_fun.integration_function_name_
                   << " must have type real[] for intial system state;"
                   << " found type="
                   << ode_fun.y0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.t0_.expression_type().is_primitive()) {
        error_msgs << "third argument to "
                   << ode_fun.integration_function_name_
                   << " must have type real or int for initial time;"
                   << " found type="
                   << ode_fun.t0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.ts_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fourth argument to "
                   << ode_fun.integration_function_name_
                   << " must have type real[]"
                   << " for requested solution times; found type="
                   << ode_fun.ts_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.theta_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fifth argument to "
                   << ode_fun.integration_function_name_
                   << " must have type real[] for parameters; found type="
                   << ode_fun.theta_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "sixth argument to "
                   << ode_fun.integration_function_name_
                   << " must have type real[] for real data; found type="
                   << ode_fun.x_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_int_.expression_type() != expr_type(INT_T, 1)) {
        error_msgs << "seventh argument to "
                   << ode_fun.integration_function_name_
                   << " must have type int[] for integer data; found type="
                   << ode_fun.x_int_.expression_type()
                   << ". ";
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(ode_fun.t0_, var_map)) {
        error_msgs << "third argument to "
                   << ode_fun.integration_function_name_
                   << " (initial times)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.ts_, var_map)) {
        error_msgs << "fourth argument to "
                   << ode_fun.integration_function_name_
                   << " (solution times)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.x_, var_map)) {
        error_msgs << "sixth argument to "
                   << ode_fun.integration_function_name_
                   << " (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
    }

    void validate_integrate_ode::operator()(const integrate_ode& ode_fun,
                                            const variable_map& var_map,
                                            bool& pass,
                                            std::ostream& error_msgs) const {
      validate_integrate_ode_non_control_args(ode_fun, var_map, pass,
                                              error_msgs);
    }
    boost::phoenix::function<validate_integrate_ode> validate_integrate_ode_f;

    void validate_integrate_ode_control::operator()(
                      const integrate_ode_control& ode_fun,
                      const variable_map& var_map, bool& pass,
                      std::ostream& error_msgs) const {
      validate_integrate_ode_non_control_args(ode_fun, var_map, pass,
                                              error_msgs);
      if (!ode_fun.rel_tol_.expression_type().is_primitive()) {
        error_msgs << "eighth argument to "
                   << ode_fun.integration_function_name_
                   << " (relative tolerance) must have type real or int;"
                   << " found type="
                   << ode_fun.rel_tol_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.abs_tol_.expression_type().is_primitive()) {
        error_msgs << "ninth argument to "
                   << ode_fun.integration_function_name_
                   << " (absolute tolerance) must have type real or int;"
                   << " found type="
                   << ode_fun.abs_tol_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.max_num_steps_.expression_type().is_primitive()) {
        error_msgs << "tenth argument to "
                   << ode_fun.integration_function_name_
                   << " (max steps) must have type real or int;"
                   << " found type="
                   << ode_fun.max_num_steps_.expression_type()
                   << ". ";
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(ode_fun.rel_tol_, var_map)) {
        error_msgs << "eight argument to "
                   << ode_fun.integration_function_name_
                   << " (relative tolerance) must be data only"
                   << " and not depend on parameters";
        pass = false;
      }
      if (has_var(ode_fun.abs_tol_, var_map)) {
        error_msgs << "ninth argument to "
                   << ode_fun.integration_function_name_
                   << " (absolute tolerance ) must be data only"
                   << " and not depend parameters";
        pass = false;
      }
      if (has_var(ode_fun.max_num_steps_, var_map)) {
        error_msgs << "tenth argument to "
                   << ode_fun.integration_function_name_
                   << " (max steps) must be data only"
                   << " and not depend on parameters";
        pass = false;
      }
    }
    boost::phoenix::function<validate_integrate_ode_control>
    validate_integrate_ode_control_f;

    template <class T>
    void validate_algebra_solver_non_control_args(const T& alg_fun,
                                                  const variable_map& var_map,
                                                  bool& pass,
                                                  std::ostream& error_msgs) {
      pass = true;
      // test function argument type
      expr_type sys_result_type(VECTOR_T, 0);
      std::vector<function_arg_type> sys_arg_types;
      sys_arg_types.push_back(function_arg_type(expr_type(VECTOR_T,
                                                          0), true));  // y
      sys_arg_types.push_back(function_arg_type(expr_type(VECTOR_T,
                                                          0)));  // theta
      sys_arg_types.push_back(function_arg_type(expr_type(DOUBLE_T,
                                                          1), true));  // x_r
      sys_arg_types.push_back(function_arg_type(expr_type(INT_T,
                                                          1)));  // x_i
      function_signature_t system_signature(sys_result_type, sys_arg_types);
      if (!function_signatures::instance()
          .is_defined(alg_fun.system_function_name_, system_signature)) {
        error_msgs << "first argument to "
                   << "algebra_solver"
                   << " must be the name of a function with signature"
                   << " (vector, vector, real[], int[]) : vector "
                   << std::endl;
        pass = false;
      }

      // test regular argument types
      if (alg_fun.y_.expression_type() != expr_type(VECTOR_T, 0)) {
        error_msgs << "second argument to algebra_solver"
                   << " must have type vector for initial guess;"
                   << " found type = "
                   << alg_fun.y_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }
      if (alg_fun.theta_.expression_type() != expr_type(VECTOR_T, 0)) {
        error_msgs << "third argument to algebra_solver"
                   << " must have type vector for parameters;"
                   << " found type = "
                   << alg_fun.theta_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }
      if (alg_fun.x_r_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fourth argument to algebra_solver"
                   << " must have type real[] for real data;"
                   << " found type = "
                   << alg_fun.x_r_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }
      if (alg_fun.x_i_.expression_type() != expr_type(INT_T, 1)) {
        error_msgs << "fifth argument to algebra_solver"
                   << " must have type int[] for integer data;"
                   << " found type = "
                   << alg_fun.x_i_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(alg_fun.y_, var_map)) {
        error_msgs << "second argument to algebra_solver"
                   << " (initial guess)"
                   << " must be data only and not reference parameters"
                   << std::endl;
        pass = false;
      }
      if (has_var(alg_fun.x_r_, var_map)) {
        error_msgs << "fourth argument to algebra_solver"
                   << " (real data)"
                   << " must be data only and not reference parameters"
                   << std::endl;
        pass = false;
      }
    }

    void validate_algebra_solver::operator()(const algebra_solver& alg_fun,
                                             const variable_map& var_map,
                                             bool& pass,
                                             std::ostream& error_msgs) const {
      validate_algebra_solver_non_control_args(alg_fun, var_map, pass,
                                               error_msgs);
    }
    boost::phoenix::function<validate_algebra_solver>
      validate_algebra_solver_f;

    void validate_algebra_solver_control::operator()(
                      const algebra_solver_control& alg_fun,
                      const variable_map& var_map, bool& pass,
                      std::ostream& error_msgs) const {
      validate_algebra_solver_non_control_args(alg_fun, var_map, pass,
                                              error_msgs);
      if (!alg_fun.rel_tol_.expression_type().is_primitive()) {
        error_msgs << "sixth argument to algebra_solver "
                   << " (relative tolerance) must have type real or int;"
                   << " found type="
                   << alg_fun.rel_tol_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }
      if (!alg_fun.fun_tol_.expression_type().is_primitive()) {
        error_msgs << "seventh argument to algebra_solver "
                   << " (function tolerance) must have type real or int;"
                   << " found type="
                   << alg_fun.fun_tol_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }
      if (!alg_fun.max_num_steps_.expression_type().is_primitive()) {
        error_msgs << "eighth argument to algebra_solver"
                   << " (max number of steps) must have type real or int;"
                   << " found type="
                   << alg_fun.max_num_steps_.expression_type()
                   << ". " << std::endl;
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(alg_fun.rel_tol_, var_map)) {
        error_msgs << "sixth argument to algebra_solver"
                   << " (relative tolerance) must be data only"
                   << " and not depend on parameters"
                   << std::endl;
        pass = false;
      }
      if (has_var(alg_fun.fun_tol_, var_map)) {
        error_msgs << "seventh argument to algebra_solver"
                   << " (function tolerance ) must be data only"
                   << " and not depend parameters"
                   << std::endl;
        pass = false;
      }
      if (has_var(alg_fun.max_num_steps_, var_map)) {
        error_msgs << "eighth argument to algebra_solver"
                   << " (max number of steps) must be data only"
                   << " and not depend on parameters" << std::endl;
        pass = false;
      }
    }
    boost::phoenix::function<validate_algebra_solver_control>
    validate_algebra_solver_control_f;


    void set_fun_type_named::operator()(expression& fun_result, fun& fun,
                                        const scope& var_scope,
                                        bool& pass,
                                        const variable_map& var_map,
                                        std::ostream& error_msgs) const {
      if (fun.name_ == "get_lp")
        error_msgs << "Warning (non-fatal): get_lp() function deprecated."
                   << std::endl
                   << "  It will be removed in a future release."
                   << std::endl
                   << "  Use target() instead."
                   << std::endl;
      if (fun.name_ == "target")
        fun.name_ = "get_lp";  // for code gen and context validation

      std::vector<expr_type> arg_types;
      for (size_t i = 0; i < fun.args_.size(); ++i)
        arg_types.push_back(fun.args_[i].expression_type());

      fun.type_ = function_signatures::instance()
        .get_result_type(fun.name_, arg_types, error_msgs);

      if (fun.type_ == ILL_FORMED_T) {
        pass = false;
        return;
      }

      // get function definition for this functiion
      std::vector<function_arg_type> fun_arg_types;
      for (size_t i = 0; i < fun.args_.size(); ++i)
        fun_arg_types.push_back(function_arg_type(arg_types[i]));
      function_signature_t sig(fun.type_, fun_arg_types);
      function_signature_t decl_sig =
        function_signatures::instance().get_definition(fun.name_, sig);
      if (!decl_sig.first.is_ill_formed()) {
        for (size_t i = 0; i < fun_arg_types.size(); ++i) {
          if (decl_sig.second[i].data_only_) {
            if (has_var(fun.args_[i], var_map)) {
              error_msgs << "Function argument error, function: "
                         << fun.name_ << ", argument: " << (i + 1)
                         << " must be data only, "
                         << "found expression containing a parameter varaible."
                         << std::endl;
              pass = false;
              return;
            }
          }
        }
      }

      // disjunction so only first match triggered
      deprecate_fun("binomial_coefficient_log", "lchoose", fun, error_msgs)
      || deprecate_fun("multiply_log", "lmultiply", fun, error_msgs)
      || deprecate_suffix("_cdf_log", "'_lcdf'", fun, error_msgs)
      || deprecate_suffix("_ccdf_log", "'_lccdf'", fun, error_msgs)
      || deprecate_suffix("_log",
              "'_lpdf' for density functions or '_lpmf' for mass functions",
              fun, error_msgs);

      // add stan::math:: qualifier for built-in nullary and math.h
      qualify_builtins(fun);
      qualify_cpp11_builtins(fun);

      // use old function names for built-in prob funs
      if (!function_signatures::instance().has_user_defined_key(fun.name_)) {
        replace_suffix("_lpdf", "_log", fun);
        replace_suffix("_lpmf", "_log", fun);
        replace_suffix("_lcdf", "_cdf_log", fun);
        replace_suffix("_lccdf", "_ccdf_log", fun);
      }
      // know these are not user-defined`x
      replace_suffix("lmultiply", "multiply_log", fun);
      replace_suffix("lchoose", "binomial_coefficient_log", fun);

      if (has_rng_suffix(fun.name_)) {
        if (!(var_scope.allows_rng())) {
          error_msgs << "ERROR: random number generators only allowed in"
                     << " transformed data block, generated quantities block"
                     << " or user-defined functions with names ending in _rng"
                     << "; found function=" << fun.name_ << " in block=";
          print_scope(error_msgs, var_scope);
          error_msgs << std::endl;
          pass = false;
          return;
        }
      }

      if (has_lp_suffix(fun.name_) || fun.name_ == "target") {
        if (!(var_scope.allows_lp_fun())) {
          error_msgs << "Function target() or functions suffixed with _lp only"
                     << " allowed in transformed parameter block, model block"
                     << std::endl
                     << "or the body of a function with suffix _lp."
                     << std::endl
                     << "Found function = "
                     << (fun.name_ == "get_lp" ? "target or get_lp" : fun.name_)
                     << " in block = ";
          print_scope(error_msgs, var_scope);
          error_msgs << std::endl;
          pass = false;
          return;
        }
      }

      if (fun.name_ == "max" || fun.name_ == "min") {
        if (fun.args_.size() == 2) {
          if (fun.args_[0].expression_type().is_primitive_int()
              && fun.args_[1].expression_type().is_primitive_int()) {
            fun.name_ = "std::" + fun.name_;
          }
        }
      }

      if (fun.name_ == "abs"
          && fun.args_.size() > 0
          && fun.args_[0].expression_type().is_primitive_double()) {
        error_msgs << "Warning: Function abs(real) is deprecated"
                   << " in the Stan language."
                   << std::endl
                   << "         It will be removed in a future release."
                   << std::endl
                   << "         Use fabs(real) instead."
                   << std::endl << std::endl;
      }

      if (fun.name_ == "lkj_cov_log") {
        error_msgs << "Warning: the lkj_cov_log() function"
                   << " is deprecated.  It will be removed in Stan 3."
                   << std::endl
                   << "Code LKJ covariance in terms of an lkj_corr()"
                   << " distribution on a correlation matrix"
                   << " and independent lognormals on the scales."
                   << std::endl << std::endl;
      }

      if (fun.name_ == "if_else") {
        error_msgs << "Warning (non-fatal): the if_else() function"
                   << " is deprecated.  "
                   << "Use the conditional operator '?:' instead."
                   << std::endl;
      }

      fun_result = fun;
      pass = true;
    }
    boost::phoenix::function<set_fun_type_named> set_fun_type_named_f;

    void infer_array_expr_type::operator()(expression& e,
                      array_expr& array_expr,
                      const scope& var_scope,
                      bool& pass,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const {
      if (array_expr.args_.size() == 0) {
        // shouldn't occur, because of % operator used to construct it
        error_msgs << "Array expression found size 0, must be > 0";
        array_expr.type_ = expr_type(ILL_FORMED_T);
        pass = false;
        return;
      }
      expr_type et;
      et = array_expr.args_[0].expression_type();
      for (size_t i = 1; i < array_expr.args_.size(); ++i) {
        expr_type et_next;
        et_next = array_expr.args_[i].expression_type();
        if (et.num_dims_ != et_next.num_dims_) {
          error_msgs << "Expressions for elements of array must have"
                     << " same array sizes; found"
                     << " previous type=" << et
                     << "; type at position " << i << "=" << et_next;
          array_expr.type_ = expr_type(ILL_FORMED_T);
          pass = false;
          return;
        }
        if ((et.base_type_ == INT_T && et_next.base_type_ == DOUBLE_T)
            || (et.base_type_ == DOUBLE_T && et_next.base_type_ == INT_T)) {
          et.base_type_ = DOUBLE_T;
        } else if (et.base_type_ != et_next.base_type_) {
          error_msgs << "Expressions for elements of array must have"
                     << " the same or promotable types; found"
                     << " previous type=" << et
                     << "; type at position " << i << "=" << et_next;
          array_expr.type_ = expr_type(ILL_FORMED_T);
          pass = false;
          return;
        }
      }
      ++et.num_dims_;
      array_expr.type_ = et;
      array_expr.array_expr_scope_ = var_scope;
      array_expr.has_var_ = has_var(array_expr, var_map);
      e = array_expr;
      pass = true;
    }
    boost::phoenix::function<infer_array_expr_type> infer_array_expr_type_f;

    void infer_vec_or_matrix_expr_type::operator()(expression& e,
                                       row_vector_expr& vec_expr,
                                       const scope& var_scope,
                                       bool& pass,
                                       const variable_map& var_map,
                                       std::ostream& error_msgs) const {
      if (vec_expr.args_.size() == 0) {
        // shouldn't occur, because of % operator used to construct it
        error_msgs << "Vector or matrix expression found size 0, must be > 0";
        pass = false;
        return;
      }
      expr_type et = vec_expr.args_[0].expression_type();
      if (!(et.is_primitive() || et.type() == ROW_VECTOR_T)) {
          error_msgs << "Matrix expression elements must be type row_vector "
                     << "and row vector expression elements must be int "
                     << "or real, but found element of type "
                     << et << std::endl;
          pass = false;
          return;
      }
      bool is_matrix = et.type() == ROW_VECTOR_T;
      for (size_t i = 1; i < vec_expr.args_.size(); ++i) {
        if (is_matrix &&
            !(vec_expr.args_[i].expression_type() == ROW_VECTOR_T)) {
          error_msgs << "Matrix expression elements must be type row_vector, "
                     << "but found element of type "
                     << vec_expr.args_[i].expression_type() << std::endl;
          pass = false;
          return;
        } else if (!(is_matrix) &&
                   !(vec_expr.args_[i].expression_type().is_primitive())) {
          error_msgs << "Row vector expression elements must be int or real, "
                     << "but found element of type "
                     << vec_expr.args_[i].expression_type() << std::endl;
          pass = false;
          return;
        }
      }
      if (is_matrix) {
        // create matrix expr object
        matrix_expr me = matrix_expr(vec_expr.args_);
        me.matrix_expr_scope_ = var_scope;
        me.has_var_ = has_var(me, var_map);
        e = me;
      } else {
        vec_expr.row_vector_expr_scope_ = var_scope;
        vec_expr.has_var_ = has_var(vec_expr, var_map);
        e = vec_expr;
      }
      pass = true;
    }
    boost::phoenix::function<infer_vec_or_matrix_expr_type>
    infer_vec_or_matrix_expr_type_f;

    void exponentiation_expr::operator()(expression& expr1,
                                         const expression& expr2,
                                         const scope& var_scope,
                                         bool& pass,
                                         std::ostream& error_msgs) const {
      if (!expr1.expression_type().is_primitive()
          || !expr2.expression_type().is_primitive()) {
        error_msgs << "arguments to ^ must be primitive (real or int)"
                   << "; cannot exponentiate "
                   << expr1.expression_type()
                   << " by "
                   << expr2.expression_type()
                   << " in block=";
        print_scope(error_msgs, var_scope);
        error_msgs << std::endl;
        pass = false;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("pow", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<exponentiation_expr> exponentiation_f;

    void multiplication_expr::operator()(expression& expr1,
                                         const expression& expr2,
                                         std::ostream& error_msgs) const {
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()) {
        expr1 *= expr2;;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("multiply", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<multiplication_expr> multiplication_f;

    void division_expr::operator()(expression& expr1,
                                   const expression& expr2,
                                   std::ostream& error_msgs) const {
      static const bool user_facing = true;
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()
          && (expr1.expression_type().is_primitive_double()
              || expr2.expression_type().is_primitive_double())) {
        expr1 /= expr2;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      if (expr1.expression_type().is_primitive_int()
          && expr2.expression_type().is_primitive_int()) {
        // result might be assigned to real - generate warning
        error_msgs << "Warning: integer division"
                   << " implicitly rounds to integer."
                   << " Found int division: ";
        generate_expression(expr1.expr_, user_facing, error_msgs);
        error_msgs << " / ";
        generate_expression(expr2.expr_, user_facing, error_msgs);
        error_msgs << std::endl
                   << " Positive values rounded down,"
                   << " negative values rounded up or down"
                   << " in platform-dependent way."
                   << std::endl;

        fun f("divide", args);
        set_fun_type(f, error_msgs);
        expr1 = expression(f);
        return;
      }
      if ((expr1.expression_type().type() == MATRIX_T
           || expr1.expression_type().type() == ROW_VECTOR_T)
          && expr2.expression_type().type() == MATRIX_T) {
        fun f("mdivide_right", args);
        set_fun_type(f, error_msgs);
        expr1 = expression(f);
        return;
      }
      fun f("divide", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
      return;
    }
    boost::phoenix::function<division_expr> division_f;

    void modulus_expr::operator()(expression& expr1, const expression& expr2,
                                  bool& pass, std::ostream& error_msgs) const {
      if (!expr1.expression_type().is_primitive_int()
          && !expr2.expression_type().is_primitive_int()) {
        error_msgs << "both operands of % must be int"
                   << "; cannot modulo "
                   << expr1.expression_type()
                   << " by "
                   << expr2.expression_type();
        error_msgs << std::endl;
        pass = false;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("modulus", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<modulus_expr> modulus_f;

    void left_division_expr::operator()(expression& expr1, bool& pass,
                                        const expression& expr2,
                                        std::ostream& error_msgs) const {
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      if (expr1.expression_type().type() == MATRIX_T
          && (expr2.expression_type().type() == VECTOR_T
              || expr2.expression_type().type() == MATRIX_T)) {
        fun f("mdivide_left", args);
        set_fun_type(f, error_msgs);
        expr1 = expression(f);
        pass = true;
        return;
      }
      fun f("mdivide_left", args);  // set for alt args err msg
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
      pass = false;
    }
    boost::phoenix::function<left_division_expr> left_division_f;

    void elt_multiplication_expr::operator()(expression& expr1,
                                             const expression& expr2,
                                             std::ostream& error_msgs) const {
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()) {
        expr1 *= expr2;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("elt_multiply", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<elt_multiplication_expr> elt_multiplication_f;

    void elt_division_expr::operator()(expression& expr1,
                                       const expression& expr2,
                                       std::ostream& error_msgs) const {
      if (expr1.expression_type().is_primitive()
          && expr2.expression_type().is_primitive()) {
        expr1 /= expr2;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      fun f("elt_divide", args);
      set_fun_type(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<elt_division_expr> elt_division_f;

    void negate_expr::operator()(expression& expr_result,
                                 const expression& expr,  bool& pass,
                                 std::ostream& error_msgs) const {
      if (expr.expression_type().is_primitive()) {
        expr_result = expression(unary_op('-', expr));
        return;
      }
      std::vector<expression> args;
      args.push_back(expr);
      fun f("minus", args);
      set_fun_type(f, error_msgs);
      expr_result = expression(f);
    }
    boost::phoenix::function<negate_expr> negate_expr_f;

    void logical_negate_expr::operator()(expression& expr_result,
                                         const expression& expr,
                                         std::ostream& error_msgs) const {
      if (!expr.expression_type().is_primitive()) {
        error_msgs << "logical negation operator !"
                   << " only applies to int or real types; ";
        expr_result = expression();
      }
      std::vector<expression> args;
      args.push_back(expr);
      fun f("logical_negation", args);
      set_fun_type(f, error_msgs);
      expr_result = expression(f);
    }
    boost::phoenix::function<logical_negate_expr> logical_negate_expr_f;

    void transpose_expr::operator()(expression& expr, bool& pass,
                                    std::ostream& error_msgs) const {
      if (expr.expression_type().is_primitive())
        return;
      std::vector<expression> args;
      args.push_back(expr);
      fun f("transpose", args);
      set_fun_type(f, error_msgs);
      expr = expression(f);
      pass = !expr.expression_type().is_ill_formed();
    }
    boost::phoenix::function<transpose_expr> transpose_f;

    void add_idxs::operator()(expression& e, std::vector<idx>& idxs,
                              bool& pass, std::ostream& error_msgs) const {
      e = index_op_sliced(e, idxs);
      pass = !e.expression_type().is_ill_formed();
      if (!pass)
        error_msgs << "Indexed expression must have at least as many"
                   << " dimensions as number of indexes supplied:"
                   << std::endl
                   << " indexed expression dims="
                   << e.total_dims()
                   << "; num indexes=" << idxs.size()
                   << std::endl;
    }
    boost::phoenix::function<add_idxs> add_idxs_f;

    void add_expression_dimss::operator()(expression& expression,
                 std::vector<std::vector<stan::lang::expression> >& dimss,
                 bool& pass, std::ostream& error_msgs) const {
      index_op iop(expression, dimss);
      int expr_dims = expression.total_dims();
      int index_dims = num_dimss(dimss);
      if (expr_dims < index_dims) {
        error_msgs << "Indexed expression must have at least as many"
                   << " dimensions as number of indexes supplied: "
                   << std::endl
                   << "    indexed expression dimensionality = " << expr_dims
                   << "; indexes supplied = " << dimss.size()
                   << std::endl;
        pass = false;
        return;
      }
      iop.infer_type();
      if (iop.type_.is_ill_formed()) {
        error_msgs << "Indexed expression must have at least as many"
                   << " dimensions as number of indexes supplied."
                   << std::endl;
        pass = false;
        return;
      }
      pass = true;
      expression = iop;
    }
    boost::phoenix::function<add_expression_dimss> add_expression_dimss_f;

    void set_var_type::operator()(variable& var_expr,
                                  expression& val, variable_map& vm,
                                  std::ostream& error_msgs, bool& pass) const {
      std::string name = var_expr.name_;
      if (name == std::string("lp__")) {
        error_msgs << std::endl
                   << "ERROR (fatal):  Use of lp__ is no longer supported."
                   << std::endl
                   << "  Use target += ... statement to increment log density."
                   << std::endl
                   << "  Use target() function to get log density."
                   << std::endl;
        pass = false;
        return;
      } else if (name == std::string("params_r__")) {
        error_msgs << std::endl << "WARNING:" << std::endl
                   << "  Direct access to params_r__ yields an inconsistent"
                   << " statistical model in isolation and no guarantee is"
                   << " made that this model will yield valid inferences."
                   << std::endl
                   << "  Moreover, access to params_r__ is unsupported"
                   << " and the variable may be removed without notice."
                   << std::endl;
      }
      pass = vm.exists(name);
      if (pass) {
        var_expr.set_type(vm.get_base_type(name), vm.get_num_dims(name));
      } else {
        error_msgs << "variable \"" << name << '"' << " does not exist."
                   << std::endl;
        return;
      }
      val = expression(var_expr);
    }
    boost::phoenix::function<set_var_type> set_var_type_f;

    void require_vbar::operator()(bool& pass, std::ostream& error_msgs) const {
      pass = false;
      error_msgs << "Probabilty functions with suffixes _lpdf, _lpmf,"
                 << " _lcdf, and _lccdf," << std::endl
                 << "require a vertical bar (|) between the first two"
                 << " arguments." << std::endl;
    }
    boost::phoenix::function<require_vbar> require_vbar_f;



    validate_no_constraints_vis::validate_no_constraints_vis(
                                               std::stringstream& error_msgs)
      : error_msgs_(error_msgs) { }

    bool validate_no_constraints_vis::operator()(const nil& /*x*/) const {
      error_msgs_ << "nil declarations not allowed";
      return false;  // fail if arises
    }
    bool validate_no_constraints_vis::operator()(const int_var_decl& x) const {
      if (x.range_.has_low() || x.range_.has_high()) {
        error_msgs_ << "require unconstrained."
                    << " found range constraint." << std::endl;
        return false;
      }
      return true;
    }
    bool validate_no_constraints_vis::operator()(const double_var_decl& x)
      const {
      if (x.range_.has_low() || x.range_.has_high()) {
        error_msgs_ << "require unconstrained."
                    << " found range constraint." << std::endl;
        return false;
      }
      return true;
    }
    bool validate_no_constraints_vis::operator()(const vector_var_decl& x)
      const {
      if (x.range_.has_low() || x.range_.has_high()) {
        error_msgs_ << "require unconstrained."
                    << " found range constraint." << std::endl;
        return false;
      }
      return true;
    }
    bool validate_no_constraints_vis::operator()(const row_vector_var_decl& x)
      const {
      if (x.range_.has_low() || x.range_.has_high()) {
        error_msgs_ << "require unconstrained."
                    << " found range constraint." << std::endl;
        return false;
      }
      return true;
    }
    bool validate_no_constraints_vis::operator()(const matrix_var_decl& x)
      const {
      if (x.range_.has_low() || x.range_.has_high()) {
        error_msgs_ << "require unconstrained."
                    << " found range constraint." << std::endl;
        return false;
      }
      return true;
    }
    bool validate_no_constraints_vis::operator()(
                                 const unit_vector_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found unit_vector." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(const simplex_var_decl& /*x*/)
      const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found simplex." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(const ordered_var_decl& /*x*/)
      const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found ordered." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(
                         const positive_ordered_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found positive_ordered." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(
                         const cholesky_factor_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found cholesky_factor." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(
                                 const cholesky_corr_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found cholesky_factor_corr." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(
                                 const cov_matrix_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found cov_matrix." << std::endl;
      return false;
    }
    bool validate_no_constraints_vis::operator()(
                                 const corr_matrix_var_decl& /*x*/) const {
      error_msgs_ << "require unconstrained variable declaration."
                  << " found corr_matrix." << std::endl;
      return false;
    }


    data_only_expression::data_only_expression(std::stringstream& error_msgs,
                                               variable_map& var_map)
      : error_msgs_(error_msgs),
        var_map_(var_map) {
    }
    bool data_only_expression::operator()(const nil& /*e*/) const {
      return true;
    }
    bool data_only_expression::operator()(const int_literal& /*x*/) const {
      return true;
    }
    bool data_only_expression::operator()(const double_literal& /*x*/) const {
      return true;
    }
    bool data_only_expression::operator()(const array_expr& x) const {
      for (size_t i = 0; i < x.args_.size(); ++i)
        if (!boost::apply_visitor(*this, x.args_[i].expr_))
          return false;
      return true;
    }
    bool data_only_expression::operator()(const matrix_expr& x) const {
      for (size_t i = 0; i < x.args_.size(); ++i)
        if (!boost::apply_visitor(*this, x.args_[i].expr_))
          return false;
      return true;
    }
    bool data_only_expression::operator()(const row_vector_expr& x) const {
      for (size_t i = 0; i < x.args_.size(); ++i)
        if (!boost::apply_visitor(*this, x.args_[i].expr_))
          return false;
      return true;
    }
    bool data_only_expression::operator()(const variable& x) const {
      scope var_scope = var_map_.get_scope(x.name_);
      bool is_data = var_scope.allows_size();
      if (!is_data) {
        error_msgs_ << "non-data variables not allowed"
                    << " in dimension declarations."
                    << std::endl
                    << "     found variable=" << x.name_
                    << "; declared in block=";
        print_scope(error_msgs_, var_scope);
        error_msgs_ << std::endl;
      }
      return is_data;
    }
    bool data_only_expression::operator()(const integrate_ode& x) const {
      return boost::apply_visitor(*this, x.y0_.expr_)
        && boost::apply_visitor(*this, x.theta_.expr_);
    }
    bool data_only_expression::operator()(const integrate_ode_control& x)
      const {
      return boost::apply_visitor(*this, x.y0_.expr_)
        && boost::apply_visitor(*this, x.theta_.expr_);
    }
    bool data_only_expression::operator()(const algebra_solver& x) const {
      return boost::apply_visitor(*this, x.theta_.expr_);
    }
    bool data_only_expression::operator()(const algebra_solver_control& x)
      const {
      return boost::apply_visitor(*this, x.theta_.expr_);
    }
    bool data_only_expression::operator()(const fun& x) const {
      for (size_t i = 0; i < x.args_.size(); ++i)
        if (!boost::apply_visitor(*this, x.args_[i].expr_))
          return false;
      return true;
    }
    bool data_only_expression::operator()(const index_op& x) const {
      if (!boost::apply_visitor(*this, x.expr_.expr_))
        return false;
      for (size_t i = 0; i < x.dimss_.size(); ++i)
        for (size_t j = 0; j < x.dimss_[i].size(); ++j)
          if (!boost::apply_visitor(*this, x.dimss_[i][j].expr_))
            return false;
      return true;
    }
    bool data_only_expression::operator()(const index_op_sliced& x) const {
      return boost::apply_visitor(*this, x.expr_.expr_);
    }
    bool data_only_expression::operator()(const conditional_op& x) const {
      return boost::apply_visitor(*this, x.cond_.expr_)
        && boost::apply_visitor(*this, x.true_val_.expr_)
        && boost::apply_visitor(*this, x.false_val_.expr_);
    }
    bool data_only_expression::operator()(const binary_op& x) const {
      return boost::apply_visitor(*this, x.left.expr_)
        && boost::apply_visitor(*this, x.right.expr_);
    }
    bool data_only_expression::operator()(const unary_op& x) const {
      return boost::apply_visitor(*this, x.subject.expr_);
    }

    void validate_decl_constraints::operator()(const bool& allow_constraints,
                                               const bool& declaration_ok,
                                               const var_decl& var_decl,
                                               bool& pass,
                                               std::stringstream& error_msgs)
      const {
      if (!declaration_ok) {
        error_msgs << "Problem with declaration." << std::endl;
        pass = false;
        return;  // short-circuits test of constraints
      }
      if (allow_constraints) {
        pass = true;
        return;
      }
      validate_no_constraints_vis vis(error_msgs);
      pass = boost::apply_visitor(vis, var_decl.decl_);
    }
    boost::phoenix::function<validate_decl_constraints>
    validate_decl_constraints_f;

    void validate_definition::operator()(const scope& var_scope,
                                         const var_decl& var_decl,
                                         bool& pass,
                                         std::stringstream& error_msgs)
      const {
      if (!var_decl.has_def()) return;

      // validate that assigment is allowed in this block
      if (!var_scope.allows_assignment()) {
        error_msgs << "variable definition not possible in this block"
                   << std::endl;
        pass = false;
      }

      // validate type
      expr_type decl_type(var_decl.base_decl().base_type_,
                          var_decl.dims().size());
      expr_type def_type = var_decl.def().expression_type();

      bool types_compatible
        = (decl_type.is_primitive()
           && def_type.is_primitive()
           && (decl_type.type() == def_type.type()
               || (decl_type.type() == DOUBLE_T
                   && def_type.type() == INT_T)))
        || (decl_type.type() == def_type.type());
      if (!types_compatible) {
        error_msgs << "variable definition base type mismatch,"
                   << " variable declared as base type: ";
        write_base_expr_type(error_msgs, decl_type.type());
        error_msgs << " variable definition has base: ";
        write_base_expr_type(error_msgs, def_type.type());
        pass = false;
      }
      // validate dims
      if (decl_type.num_dims() != def_type.num_dims()) {
        error_msgs << "variable definition dimensions mismatch,"
                   << " definition specifies "
                   <<  decl_type.num_dims()
                   << ", declaration specifies "
                   << def_type.num_dims();
        pass = false;
      }
      return;
    }
    boost::phoenix::function<validate_definition>
    validate_definition_f;


    void validate_identifier::reserve(const std::string& w) {
      reserved_word_set_.insert(w);
    }
    bool validate_identifier::contains(const std::set<std::string>& s,
                                       const std::string& x) const {
      return s.find(x) != s.end();
    }
    bool validate_identifier::identifier_exists(const std::string& identifier)
      const {
      return contains(reserved_word_set_, identifier)
        || (contains(function_signatures::instance().key_set(), identifier)
            && !contains(const_fun_name_set_, identifier));
    }

    validate_identifier::validate_identifier() {
      // constant functions which may be used as identifiers
      const_fun_name_set_.insert("pi");
      const_fun_name_set_.insert("e");
      const_fun_name_set_.insert("sqrt2");
      const_fun_name_set_.insert("log2");
      const_fun_name_set_.insert("log10");
      const_fun_name_set_.insert("not_a_number");
      const_fun_name_set_.insert("positive_infinity");
      const_fun_name_set_.insert("negative_infinity");
      const_fun_name_set_.insert("epsilon");
      const_fun_name_set_.insert("negative_epsilon");
      const_fun_name_set_.insert("machine_precision");

      // illegal identifiers
      reserve("for");
      reserve("in");
      reserve("while");
      reserve("repeat");
      reserve("until");
      reserve("if");
      reserve("then");
      reserve("else");
      reserve("true");
      reserve("false");

      reserve("int");
      reserve("real");
      reserve("vector");
      reserve("unit_vector");
      reserve("simplex");
      reserve("ordered");
      reserve("positive_ordered");
      reserve("row_vector");
      reserve("matrix");
      reserve("cholesky_factor_cov");
      reserve("cholesky_factor_corr");
      reserve("cov_matrix");
      reserve("corr_matrix");

      reserve("target");

      reserve("model");
      reserve("data");
      reserve("parameters");
      reserve("quantities");
      reserve("transformed");
      reserve("generated");

      reserve("var");
      reserve("fvar");
      reserve("STAN_MAJOR");
      reserve("STAN_MINOR");
      reserve("STAN_PATCH");
      reserve("STAN_MATH_MAJOR");
      reserve("STAN_MATH_MINOR");
      reserve("STAN_MATH_PATCH");

      reserve("alignas");
      reserve("alignof");
      reserve("and");
      reserve("and_eq");
      reserve("asm");
      reserve("auto");
      reserve("bitand");
      reserve("bitor");
      reserve("bool");
      reserve("break");
      reserve("case");
      reserve("catch");
      reserve("char");
      reserve("char16_t");
      reserve("char32_t");
      reserve("class");
      reserve("compl");
      reserve("const");
      reserve("constexpr");
      reserve("const_cast");
      reserve("continue");
      reserve("decltype");
      reserve("default");
      reserve("delete");
      reserve("do");
      reserve("double");
      reserve("dynamic_cast");
      reserve("else");
      reserve("enum");
      reserve("explicit");
      reserve("export");
      reserve("extern");
      reserve("false");
      reserve("float");
      reserve("for");
      reserve("friend");
      reserve("goto");
      reserve("if");
      reserve("inline");
      reserve("int");
      reserve("long");
      reserve("mutable");
      reserve("namespace");
      reserve("new");
      reserve("noexcept");
      reserve("not");
      reserve("not_eq");
      reserve("nullptr");
      reserve("operator");
      reserve("or");
      reserve("or_eq");
      reserve("private");
      reserve("protected");
      reserve("public");
      reserve("register");
      reserve("reinterpret_cast");
      reserve("return");
      reserve("short");
      reserve("signed");
      reserve("sizeof");
      reserve("static");
      reserve("static_assert");
      reserve("static_cast");
      reserve("struct");
      reserve("switch");
      reserve("template");
      reserve("this");
      reserve("thread_local");
      reserve("throw");
      reserve("true");
      reserve("try");
      reserve("typedef");
      reserve("typeid");
      reserve("typename");
      reserve("union");
      reserve("unsigned");
      reserve("using");
      reserve("virtual");
      reserve("void");
      reserve("volatile");
      reserve("wchar_t");
      reserve("while");
      reserve("xor");
      reserve("xor_eq");

      // function names declared in signatures
      using stan::lang::function_signatures;
      using std::set;
      using std::string;
      const function_signatures& sigs = function_signatures::instance();

      set<string> fun_names = sigs.key_set();
      for (set<string>::iterator it = fun_names.begin();
           it != fun_names.end();
           ++it)
        if (!contains(const_fun_name_set_, *it))
          reserve(*it);
    }

    void validate_identifier::operator()(const std::string& identifier,
                                         bool& pass,
                                         std::stringstream& error_msgs) const {
      int len = identifier.size();
      if (len >= 2
          && identifier[len-1] == '_'
          && identifier[len-2] == '_') {
        error_msgs << "variable identifier (name) may"
                   << " not end in double underscore (__)"
                   << std::endl
                   << "    found identifer=" << identifier << std::endl;
        pass = false;
        return;
      }
      size_t period_position = identifier.find('.');
      if (period_position != std::string::npos) {
        error_msgs << "variable identifier may not contain a period (.)"
                   << std::endl
                   << "    found period at position (indexed from 0)="
                   << period_position
                   << std::endl
                   << "    found identifier=" << identifier
                   << std::endl;
        pass = false;
        return;
      }
      if (identifier_exists(identifier)) {
        error_msgs << "variable identifier (name) may not be reserved word"
                   << std::endl
                   << "    found identifier=" << identifier
                   << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_identifier> validate_identifier_f;

    // copies single dimension from M to N if only M declared
    void copy_square_cholesky_dimension_if_necessary::operator()(
                           cholesky_factor_var_decl& var_decl) const {
        if (is_nil(var_decl.N_))
          var_decl.N_ = var_decl.M_;
    }
    boost::phoenix::function<copy_square_cholesky_dimension_if_necessary>
    copy_square_cholesky_dimension_if_necessary_f;

    void empty_range::operator()(range& r,
                                 std::stringstream& /*error_msgs*/) const {
      r = range();
    }
    boost::phoenix::function<empty_range> empty_range_f;

    void set_int_range_lower::operator()(range& range,
                                         const expression& expr,
                                         bool& pass,
                                         std::stringstream& error_msgs) const {
      range.low_ = expr;
      validate_int_expr validator;
      validator(expr, pass, error_msgs);
    }
    boost::phoenix::function<set_int_range_lower> set_int_range_lower_f;

    void set_int_range_upper::operator()(range& range,
                                         const expression& expr,
                                         bool& pass,
                                         std::stringstream& error_msgs) const {
      range.high_ = expr;
      validate_int_expr validator;
      validator(expr, pass, error_msgs);
    }
    boost::phoenix::function<set_int_range_upper> set_int_range_upper_f;

    void validate_int_data_expr::operator()(const expression& expr,
                                            const scope& var_scope,
                                            bool& pass,
                                            variable_map& var_map,
                                            std::stringstream& error_msgs)
      const {
      if (!expr.expression_type().is_primitive_int()) {
        error_msgs << "dimension declaration requires expression"
                   << " denoting integer; found type="
                   << expr.expression_type()
                   << std::endl;
        pass = false;
        return;
      }

      if (!var_scope.is_local()) {
        data_only_expression vis(error_msgs, var_map);
        bool only_data_dimensions = boost::apply_visitor(vis, expr.expr_);
        pass = only_data_dimensions;
        return;
      }

      // don't need to check data vs. parameter in dimensions for
      // local variable declarations
      pass = true;
    }
    boost::phoenix::function<validate_int_data_expr> validate_int_data_expr_f;

    void set_double_range_lower::operator()(range& range,
                                            const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      range.low_ = expr;
      validate_double_expr validator;
      validator(expr, pass, error_msgs);
    }
    boost::phoenix::function<set_double_range_lower> set_double_range_lower_f;

    void set_double_range_upper::operator()(range& range,
                                            const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      range.high_ = expr;
      validate_double_expr validator;
      validator(expr, pass, error_msgs);
    }
    boost::phoenix::function<set_double_range_upper> set_double_range_upper_f;

    template <typename T>
    void add_var::operator()(var_decl& var_decl_result, const T& var_decl,
                             variable_map& vm, bool& pass,
                             const scope& var_scope,
                             std::ostream& error_msgs) const {
      if (vm.exists(var_decl.name_)) {
        pass = false;
        error_msgs << "duplicate declaration of variable, name="
                   << var_decl.name_;

        error_msgs << "; attempt to redeclare as ";
        print_scope(error_msgs, var_scope);

        error_msgs << "; original declaration as ";
        print_scope(error_msgs, vm.get_scope(var_decl.name_));

        error_msgs << std::endl;
        var_decl_result = var_decl;
        return;
      }
      if (var_scope.par_or_tpar()
           && var_decl.base_type_ == INT_T) {
        pass = false;
        error_msgs << "parameters or transformed parameters"
                   << " cannot be integer or integer array; "
                   << " found declared type int, parameter name="
                   << var_decl.name_
                   << std::endl;
        var_decl_result = var_decl;
        return;
      }
      pass = true;
      vm.add(var_decl.name_, var_decl, var_scope);
      var_decl_result = var_decl;
    }
    boost::phoenix::function<add_var> add_var_f;

    template void add_var::operator()(var_decl&, const int_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const double_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const vector_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const row_vector_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const matrix_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const simplex_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const unit_vector_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const ordered_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&,
                                      const positive_ordered_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&,
                                      const cholesky_factor_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const cholesky_corr_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const cov_matrix_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const corr_matrix_var_decl&,
                                      variable_map&, bool&, const scope&,
                                      std::ostream&) const;

    void validate_in_loop::operator()(bool in_loop, bool& pass,
                                      std::ostream& error_msgs) const {
      pass = in_loop;
      if (!pass)
        error_msgs << "ERROR: break and continue statements are only allowed"
                   << " in the body of a for-loop or while-loop."
                   << std::endl;
    }
    boost::phoenix::function<validate_in_loop> validate_in_loop_f;

    void non_void_expression::operator()(const expression& e, bool& pass,
                                         std::ostream& error_msgs) const {
      // ill-formed shouldn't be possible, but just in case
      pass = e.expression_type().type() != VOID_T
        && e.expression_type().type() != ILL_FORMED_T;
      if (!pass)
        error_msgs << "ERROR:  expected printable (non-void) expression."
                   << std::endl;
    }
    boost::phoenix::function<non_void_expression> non_void_expression_f;

    void set_var_scope::operator()(scope& var_scope,
                                    const origin_block& program_block)
      const {
      var_scope = scope(program_block);
    }
    boost::phoenix::function<set_var_scope> set_var_scope_f;

    void set_var_scope_local::operator()(scope& var_scope,
                                         const origin_block& program_block)
      const {
      var_scope = scope(program_block, true);
    }
    boost::phoenix::function<set_var_scope_local> set_var_scope_local_f;

    void reset_var_scope::operator()(scope& var_scope,
                                     const scope& scope_enclosing)
      const {
      origin_block enclosing_block = scope_enclosing.program_block();
      var_scope = scope(enclosing_block, true);
    }
    boost::phoenix::function<reset_var_scope> reset_var_scope_f;

    void trace::operator()(const std::string& msg) const {
    }
    boost::phoenix::function<trace> trace_f;

    void deprecate_pound_comment::operator()(std::ostream& error_msgs) const {
      error_msgs << "Warning (non-fatal): Comments beginning with #"
                 << " are deprecated.  Please use // in place of #"
                 << " for line comments." << std::endl;
    }
    boost::phoenix::function<deprecate_pound_comment>
    deprecate_pound_comment_f;

  }
}

#endif
