#ifndef STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_DEF_CPP
#define STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_DEF_CPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
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

    int num_dimss(std::vector<std::vector<stan::lang::expression> >& dimss) {
      int sum = 0;
      for (size_t i = 0; i < dimss.size(); ++i)
        sum += dimss[i].size();
      return sum;
    }

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
                                         const integrate_ode_cvode&)
      const;
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

    fun set_fun_type::operator()(fun& fun,
                                 std::ostream& error_msgs) const {
      std::vector<expr_type> arg_types;
      for (size_t i = 0; i < fun.args_.size(); ++i)
        arg_types.push_back(fun.args_[i].expression_type());
      fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                  arg_types,
                                                                  error_msgs);
      return fun;
    }
    boost::phoenix::function<set_fun_type> set_fun_type_f;

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
      set_fun_type sft;
      fun f("add", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("subtract", args);
      sft(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<subtraction_expr3> subtraction3_f;

    void increment_size_t::operator()(size_t& lhs) const {
      ++lhs;
    }
    boost::phoenix::function<increment_size_t> increment_size_t_f;

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
      set_fun_type sft;
      fun f(fun_name, args);
      sft(f, error_msgs);
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
                                        var_origin& origin, bool& pass,
                                        std::ostream& error_msgs) const {
      if (return_type.is_void() && return_type.num_dims() > 0) {
        error_msgs << "Void return type may not have dimensions declared."
                   << std::endl;
        pass = false;
        return;
      }
      origin = return_type.is_void()
        ? void_function_argument_origin
        : function_argument_origin;
    }
    boost::phoenix::function<set_void_function> set_void_function_f;

    void set_allows_sampling_origin::operator()(const std::string& identifier,
                                                bool& allow_sampling,
                                                int& origin) const {
      bool is_void_function_origin
        = (origin == void_function_argument_origin);
      if (ends_with("_lp", identifier)) {
        allow_sampling = true;
        origin = is_void_function_origin
          ? void_function_argument_origin_lp
          : function_argument_origin_lp;
      } else if (ends_with("_rng", identifier)) {
        allow_sampling = false;
        origin = is_void_function_origin
          ? void_function_argument_origin_rng
          : function_argument_origin_rng;
      } else {
        allow_sampling = false;
        origin = is_void_function_origin
          ? void_function_argument_origin
          : function_argument_origin;
      }
    }
    boost::phoenix::function<set_allows_sampling_origin>
    set_allows_sampling_origin_f;

    void validate_declarations::operator()(bool& pass,
                       std::set<std::pair<std::string,
                                          function_signature_t> >& declared,
                       std::set<std::pair<std::string,
                                          function_signature_t> >& defined,
                       std::ostream& error_msgs) const {
      using std::set;
      using std::string;
      using std::pair;
      typedef set<pair<string, function_signature_t> >::iterator iterator_t;
      for (iterator_t it = declared.begin(); it != declared.end(); ++it) {
        if (defined.find(*it) == defined.end()) {
          error_msgs <<"Function declared, but not defined."
                     << " Function name=" << (*it).first
                     << std::endl;
          pass = false;
          return;
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
           ++it)
        if (name_sig.first == (*it).first
            && (name_only
                || name_sig.second.second == (*it).second.second))
          return true;  // name and arg sequences match
      return false;
    }

    void add_function_signature::operator()(const function_decl_def& decl,
              bool& pass,
              std::set<std::pair<std::string,
                                 function_signature_t> >& functions_declared,
              std::set<std::pair<std::string,
                                 function_signature_t> >& functions_defined,
              std::ostream& error_msgs) const {
      // build up representations
      expr_type result_type(decl.return_type_.base_type_,
                            decl.return_type_.num_dims_);
      std::vector<expr_type> arg_types;
      for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
        arg_types.push_back(
                            expr_type(decl.arg_decls_[i].arg_type_.base_type_,
                                      decl.arg_decls_[i].arg_type_.num_dims_));
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

      // add declaration in local sets and in parser function sigs
      if (functions_declared.find(name_sig) == functions_declared.end()) {
        functions_declared.insert(name_sig);
        function_signatures::instance()
          .add(decl.name_,
               result_type, arg_types);
        function_signatures::instance()
          .set_user_defined(name_sig);
      }

      // add as definition if there's a body
      if (!decl.body_.is_no_op_statement())
        functions_defined.insert(name_sig);
      pass = true;
    }
    boost::phoenix::function<add_function_signature> add_function_signature_f;


    void validate_return_type::operator()(function_decl_def& decl,
                                          bool& pass,
                                          std::ostream& error_msgs) const {
      pass = decl.body_.is_no_op_statement()
        || stan::lang::returns_type(decl.return_type_, decl.body_,
                                    error_msgs);
      if (!pass) {
        error_msgs << "Improper return in body of function.";
        return;
      }

      if (ends_with("_log", decl.name_)
          && !decl.return_type_.is_primitive_double()) {
        pass = false;
        error_msgs << "Require real return type for functions"
                   << " ending in _log.";
      }
    }
    boost::phoenix::function<validate_return_type> validate_return_type_f;

    void scope_lp::operator()(variable_map& vm) const {
      vm.add("lp__", DOUBLE_T, local_origin);
      vm.add("params_r__", VECTOR_T, local_origin);
    }
    boost::phoenix::function<scope_lp> scope_lp_f;

    void unscope_variables::operator()(function_decl_def& decl,
                                       variable_map& vm) const {
        vm.remove("lp__");
        vm.remove("params_r__");
        for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
          vm.remove(decl.arg_decls_[i].name_);
    }
    boost::phoenix::function<unscope_variables> unscope_variables_f;

    void add_fun_var::operator()(arg_decl& decl, bool& pass, variable_map& vm,
                                 std::ostream& error_msgs) const {
      if (vm.exists(decl.name_)) {
        // variable already exists
        pass = false;
        error_msgs << "duplicate declaration of variable, name="
                   << decl.name_
                   << "; attempt to redeclare as function argument"
                   << "; original declaration as ";
        print_var_origin(error_msgs, vm.get_origin(decl.name_));
        error_msgs << std::endl;
        return;
      }
      pass = true;
      vm.add(decl.name_, decl.base_variable_declaration(),
             function_argument_origin);
    }
    boost::phoenix::function<add_fun_var> add_fun_var_f;

    // TODO(carpenter): seems redundant; see if it can be removed
    void set_omni_idx::operator()(omni_idx& val) const {
      val = omni_idx();
    }
    boost::phoenix::function<set_omni_idx> set_omni_idx_f;

    void validate_int_expression::operator()(const expression & e, bool& pass)
      const {
      pass = e.expression_type().is_primitive_int();
    }
    boost::phoenix::function<validate_int_expression>
    validate_int_expression_f;

    void validate_ints_expression::operator()(const expression & e, bool& pass,
                                              std::ostream& error_msgs) const {
      if (e.expression_type().type() != INT_T) {
        error_msgs << "index must be integer; found type=";
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


    void add_lp_var::operator()(variable_map& vm) const {
      vm.add("lp__",
             base_var_decl("lp__", std::vector<expression>(), DOUBLE_T),
             local_origin);  // lp acts as a local where defined
      vm.add("params_r__",
             base_var_decl("params_r__", std::vector<expression>(), VECTOR_T),
             local_origin);  // lp acts as a local where defined
    }
    boost::phoenix::function<add_lp_var> add_lp_var_f;

    void remove_lp_var::operator()(variable_map& vm) const {
      vm.remove("lp__");
      vm.remove("params_r__");
    }
    boost::phoenix::function<remove_lp_var> remove_lp_var_f;

    void program_error::operator()(pos_iterator_t _begin, pos_iterator_t _end,
                                   pos_iterator_t _where, variable_map& vm,
                                   std::stringstream& error_msgs) const {
      using boost::spirit::get_line;
      using boost::format;
      using std::setw;

      size_t idx_errline = get_line(_where);

      error_msgs << std::endl;

      if (idx_errline > 0) {
        error_msgs << "ERROR at line " << idx_errline
                   << std::endl << std::endl;

        std::basic_stringstream<char> sprogram;
        sprogram << boost::make_iterator_range(_begin, _end);

        // show error in context 2 lines before, 1 lines after
        size_t idx_errcol = 0;
        idx_errcol = get_column(_begin, _where) - 1;

        std::string lineno = "";
        format fmt_lineno("% 3d:    ");

        std::string line_2before = "";
        std::string line_before = "";
        std::string line_err = "";
        std::string line_after = "";

        size_t idx_line = 0;
        size_t idx_before = idx_errline - 1;
        if (idx_before > 0) {
          // read lines up to error line, save 2 most recently read
          while (idx_before > idx_line) {
            line_2before = line_before;
            std::getline(sprogram, line_before);
            idx_line++;
          }
          if (line_2before.length() > 0) {
            lineno = str(fmt_lineno % (idx_before - 1) );
            error_msgs << lineno << line_2before << std::endl;
          }
          lineno = str(fmt_lineno % idx_before);
          error_msgs << lineno << line_before << std::endl;
        }

        std::getline(sprogram, line_err);
        lineno = str(fmt_lineno % idx_errline);
        error_msgs << lineno << line_err << std::endl
                   << setw(idx_errcol + lineno.length()) << "^" << std::endl;

        if (!sprogram.eof()) {
          std::getline(sprogram, line_after);
          lineno = str(fmt_lineno % (idx_errline+1));
          error_msgs << lineno << line_after << std::endl;
        }
      }
      error_msgs << std::endl;
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

    void validate_return_allowed::operator()(var_origin origin, bool& pass,
                                             std::ostream& error_msgs) const {
      if (origin != function_argument_origin
          && origin != function_argument_origin_lp
          && origin != function_argument_origin_rng) {
        error_msgs << "Returns only allowed from function bodies."
                   << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_return_allowed> validate_return_allowed_f;

    void validate_void_return_allowed::operator()(var_origin origin,
                                                  bool& pass,
                                                  std::ostream& error_msgs)
      const {
      if (origin != void_function_argument_origin
          && origin != void_function_argument_origin_lp
          && origin != void_function_argument_origin_rng) {
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

    void identifier_to_var::operator()(const std::string& name,
                                       const var_origin& origin_allowed,
                                       variable& v,  bool& pass,
                                       const variable_map& vm,
                                       std::ostream& error_msgs) const {
      // validate existence
      if (!vm.exists(name)) {
        pass = false;
        return;
      }
      // validate origin
      var_origin lhs_origin = vm.get_origin(name);
      if (lhs_origin != local_origin
          && lhs_origin != origin_allowed) {
        pass = false;
        return;
      }
      // enforce constancy of function args
      if (lhs_origin == function_argument_origin
          || lhs_origin == function_argument_origin_lp
          || lhs_origin == function_argument_origin_rng
          || lhs_origin == void_function_argument_origin
          || lhs_origin == void_function_argument_origin_lp
          || lhs_origin == void_function_argument_origin_rng) {
        pass = false;
        return;
      }
      v = variable(name);
      v.set_type(vm.get_base_type(name), vm.get_num_dims(name));
      pass = true;
    }
    boost::phoenix::function<identifier_to_var> identifier_to_var_f;

    void validate_assgn::operator()(const assgn& a, bool& pass,
                                    std::ostream& error_msgs) const {
      // resolve type of lhs[idxs] and make sure it matches rhs
      std::string name = a.lhs_var_.name_;
      expression lhs_expr = expression(a.lhs_var_);
      expr_type lhs_type = indexed_type(lhs_expr, a.idxs_);
      if (lhs_type.is_ill_formed()) {
        error_msgs << "Left-hand side indexing incompatible with variable."
                   << std::endl;
        pass = false;
        return;
      }

      expr_type rhs_type = a.rhs_.expression_type();
      base_expr_type lhs_base_type = lhs_type.base_type_;
      base_expr_type rhs_base_type = rhs_type.base_type_;
      // allow int -> double promotion, even in arrays
      bool types_compatible
        = lhs_base_type == rhs_base_type
        || (lhs_base_type == DOUBLE_T && rhs_base_type == INT_T);
      if (!types_compatible) {
        error_msgs << "base type mismatch in assignment"
                   << "; variable name="
                   << name
                   << ", type=";
        write_base_expr_type(error_msgs, lhs_base_type);
        error_msgs << "; right-hand side type=";
        write_base_expr_type(error_msgs, rhs_base_type);
        error_msgs << std::endl;
        pass = false;
        return;
      }

      if (lhs_type.num_dims_ != rhs_type.num_dims_) {
        error_msgs << "dimension mismatch in assignment"
                   << "; variable name="
                   << name
                   << ", num dimensions given="
                   << lhs_type.num_dims_
                   << "; right-hand side dimensions="
                   << rhs_type.num_dims_
                   << std::endl;
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
                                         const var_origin& origin_allowed,
                                         bool& pass, variable_map& vm,
                                         std::ostream& error_msgs) const {
      // validate existence
      std::string name = a.var_dims_.name_;
      if (!vm.exists(name)) {
        error_msgs << "unknown variable in assignment"
                   << "; lhs variable=" << a.var_dims_.name_
                   << std::endl;

        pass = false;
        return;
      }

      // validate origin
      var_origin lhs_origin = vm.get_origin(name);
      if (lhs_origin != local_origin
          && lhs_origin != origin_allowed) {
        error_msgs << "attempt to assign variable in wrong block."
                   << " left-hand-side variable origin=";
        print_var_origin(error_msgs, lhs_origin);
        error_msgs << std::endl;
        pass = false;
        return;
      }

      // enforce constancy of function args
      if (lhs_origin == function_argument_origin
          || lhs_origin == function_argument_origin_lp
          || lhs_origin == function_argument_origin_rng
          || lhs_origin == void_function_argument_origin
          || lhs_origin == void_function_argument_origin_lp
          || lhs_origin == void_function_argument_origin_rng) {
        error_msgs << "Illegal to assign to function argument variables."
                   << std::endl
                   << "Use local variables instead."
                   << std::endl;
        pass = false;
        return;
      }

      // validate types
      a.var_type_ = vm.get(name);
      size_t lhs_var_num_dims = a.var_type_.dims_.size();
      size_t num_index_dims = a.var_dims_.dims_.size();

      expr_type lhs_type = infer_type_indexing(a.var_type_.base_type_,
                                               lhs_var_num_dims,
                                               num_index_dims);

      if (lhs_type.is_ill_formed()) {
        error_msgs << "too many indexes for variable "
                   << "; variable name = " << name
                   << "; num dimensions given = " << num_index_dims
                   << "; variable array dimensions = " << lhs_var_num_dims
                   << std::endl;
        pass = false;
        return;
      }

      base_expr_type lhs_base_type = lhs_type.base_type_;
      base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
      // allow int -> double promotion
      bool types_compatible
        = lhs_base_type == rhs_base_type
        || (lhs_base_type == DOUBLE_T && rhs_base_type == INT_T);
      if (!types_compatible) {
        error_msgs << "base type mismatch in assignment"
                   << "; variable name = "
                   << a.var_dims_.name_
                   << ", type = ";
        write_base_expr_type(error_msgs, lhs_base_type);
        error_msgs << "; right-hand side type=";
        write_base_expr_type(error_msgs, rhs_base_type);
        error_msgs << std::endl;
        pass = false;
        return;
      }

      if (lhs_type.num_dims_ != a.expr_.expression_type().num_dims_) {
        error_msgs << "dimension mismatch in assignment"
                   << "; variable name = "
                   << a.var_dims_.name_
                   << ", num dimensions given = "
                   << lhs_type.num_dims_
                   << "; right-hand side dimensions = "
                   << a.expr_.expression_type().num_dims_
                   << std::endl;
        pass = false;
        return;
      }

      pass = true;
    }
    boost::phoenix::function<validate_assignment> validate_assignment_f;


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

    void validate_sample::operator()(const sample& s,
                                     const variable_map& var_map, bool& pass,
                                     std::ostream& error_msgs) const {
      static const bool user_facing = true;
      std::vector<expr_type> arg_types;
      arg_types.push_back(s.expr_.expression_type());
      for (size_t i = 0; i < s.dist_.args_.size(); ++i)
        arg_types.push_back(s.dist_.args_[i].expression_type());
      std::string function_name(s.dist_.family_);
      std::string internal_function_name = function_name + "_log";

      if ((internal_function_name.find("multiply_log")
           != std::string::npos)
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

      if (!is_double_return(internal_function_name, arg_types, error_msgs)) {
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

      // test for LHS not being purely a variable
      if (has_non_param_var(s.expr_, var_map)) {
        error_msgs << "Warning (non-fatal):"
                   << std::endl
                   << "Left-hand side of sampling statement (~) may contain a"
                   << " non-linear transform of a parameter or local variable."
                   << std::endl
                   << "If so, you need to call increment_log_prob() with"
                   << " the log absolute determinant of the Jacobian of"
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

      if (s.truncation_.has_low()) {
        std::vector<expr_type> arg_types_trunc(arg_types);
        arg_types_trunc[0] = s.truncation_.low_.expression_type();
        std::string function_name_cdf(s.dist_.family_);
        function_name_cdf += "_cdf_log";
        if (!is_double_return(function_name_cdf, arg_types_trunc,
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
      if (s.truncation_.has_high()) {
        std::vector<expr_type> arg_types_trunc(arg_types);
        arg_types_trunc[0] = s.truncation_.high_.expression_type();
        std::string function_name_cdf(s.dist_.family_);
        function_name_cdf += "_cdf_log";
        if (!is_double_return(function_name_cdf, arg_types_trunc,
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
                                         bool& pass, variable_map& vm,
                                         std::stringstream& error_msgs) const {
      name_local = name;
      pass = !vm.exists(name);
      if (!pass)
        error_msgs << "ERROR: loop variable already declared."
                   << " variable name=\"" << name << "\"" << std::endl;
      else
        vm.add(name, base_var_decl(name, std::vector<expression>(), INT_T),
               local_origin);  // loop var acts like local
    }
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    void remove_loop_identifier::operator()(const std::string& name,
                                            variable_map& vm) const {
      vm.remove(name);
    }
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;

    void validate_int_expr_warn::operator()(const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      pass = expr.expression_type().is_primitive_int();
      if (!pass)
        error_msgs << "expression denoting integer required; found type="
                   << expr.expression_type() << std::endl;
    }
    boost::phoenix::function<validate_int_expr_warn> validate_int_expr_warn_f;

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

    void validate_allow_sample::operator()(const bool& allow_sample,
                                           bool& pass,
                                           std::stringstream& error_msgs)
      const {
      pass = allow_sample;
      if (!pass)
        error_msgs << "Sampling statements (~) and increment_log_prob() are"
                   << std::endl
                   << "only allowed in the model block."
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


    void add_line_number::operator()(statement& stmt,
                                     const pos_iterator_t& begin,
                                     const pos_iterator_t& end) const {
      stmt.begin_line_ = get_line(begin);
      stmt.end_line_ = get_line(end);
    }
    boost::phoenix::function<add_line_number> add_line_number_f;

    void set_void_return::operator()(return_statement& s) const {
      s = return_statement();
    }
    boost::phoenix::function<set_void_return> set_void_return_f;

    void set_no_op::operator()(no_op_statement& s) const {
      s = no_op_statement();
    }
    boost::phoenix::function<set_no_op> set_no_op_f;



    void validate_integrate_ode::operator()(const integrate_ode& ode_fun,
                                            const variable_map& var_map,
                                            bool& pass,
                                            std::ostream& error_msgs) const {
      pass = true;

      // test function argument type
      expr_type sys_result_type(DOUBLE_T, 1);
      std::vector<expr_type> sys_arg_types;
      sys_arg_types.push_back(expr_type(DOUBLE_T, 0));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(INT_T, 1));
      function_signature_t system_signature(sys_result_type, sys_arg_types);
      if (!function_signatures::instance()
          .is_defined(ode_fun.system_function_name_, system_signature)) {
        error_msgs << "first argument to integrate_ode"
                   << " must be a function with signature"
                   << " (real, real[], real[], real[], int[]) : real[] ";
        pass = false;
      }

      // test regular argument types
      if (ode_fun.y0_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "second argument to integrate_ode must be type real[]"
                   << " for intial system state"
                   << "; found type="
                   << ode_fun.y0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.t0_.expression_type().is_primitive()) {
        error_msgs << "third argument to integrate_ode"
                   << " must be type real or int"
                   << " for initial time"
                   << "; found type="
                   << ode_fun.t0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.ts_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fourth argument to integrate_ode must be type real[]"
                   << " for requested solution times"
                   << "; found type="
                   << ode_fun.ts_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.theta_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fifth argument to integrate_ode must be type real[]"
                   << " for parameters"
                   << "; found type="
                   << ode_fun.theta_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "sixth argument to integrate_ode must be type real[]"
                   << " for real data;"
                   << " found type="
                   << ode_fun.x_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_int_.expression_type() != expr_type(INT_T, 1)) {
        error_msgs << "seventh argument to integrate_ode must be type int[]"
                   << " for integer data;"
                   << " found type="
                   << ode_fun.x_int_.expression_type()
                   << ". ";
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(ode_fun.t0_, var_map)) {
        error_msgs << "third argument to integrate_ode (initial times)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.ts_, var_map)) {
        error_msgs << "fourth argument to integrate_ode (solution times)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.x_, var_map)) {
        error_msgs << "fifth argument to integrate_ode (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
    }
    boost::phoenix::function<validate_integrate_ode> validate_integrate_ode_f;

    void validate_integrate_ode_cvode::operator()(
                      const integrate_ode_cvode& ode_fun,
                      const variable_map& var_map, bool& pass,
                      std::ostream& error_msgs) const {
      pass = true;

      // test function argument type
      expr_type sys_result_type(DOUBLE_T, 1);
      std::vector<expr_type> sys_arg_types;
      sys_arg_types.push_back(expr_type(DOUBLE_T, 0));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(DOUBLE_T, 1));
      sys_arg_types.push_back(expr_type(INT_T, 1));
      function_signature_t system_signature(sys_result_type, sys_arg_types);
      if (!function_signatures::instance()
          .is_defined(ode_fun.system_function_name_, system_signature)) {
        error_msgs << "first argument to integrate_ode_cvode"
                   << " must be a function with signature"
                   << " (real, real[], real[], real[], int[]) : real[] ";
        pass = false;
      }

      // test regular argument types
      if (ode_fun.y0_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "second argument to integrate_ode_cvode must be"
                   << " type real[] for intial system state"
                   << "; found type="
                   << ode_fun.y0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.t0_.expression_type().is_primitive()) {
        error_msgs << "third argument to integrate_ode_cvode"
                   << " must be type real or int"
                   << " for initial time"
                   << "; found type="
                   << ode_fun.t0_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.ts_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fourth argument to integrate_ode_cvode must be"
                   << " type real[] for requested solution times"
                   << "; found type="
                   << ode_fun.ts_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.theta_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "fifth argument to integrate_ode_cvode must be"
                   << " type real[] for parameters"
                   << "; found type="
                   << ode_fun.theta_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_.expression_type() != expr_type(DOUBLE_T, 1)) {
        error_msgs << "sixth argument to integrate_ode_cvode must be"
                   << " type real[] for real data;"
                   << " found type="
                   << ode_fun.x_.expression_type()
                   << ". ";
        pass = false;
      }
      if (ode_fun.x_int_.expression_type() != expr_type(INT_T, 1)) {
        error_msgs << "seventh argument to integrate_ode_cvode must be"
                   << " type int[] for integer data;"
                   << " found type="
                   << ode_fun.x_int_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.rel_tol_.expression_type().is_primitive()) {
        error_msgs << "eight argument to integrate_ode_cvode"
                   << " must be type real or int"
                   << " for relative tolerance"
                   << "; found type="
                   << ode_fun.rel_tol_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.abs_tol_.expression_type().is_primitive()) {
        error_msgs << "ninth argument to integrate_ode_cvode"
                   << " must be type real or int"
                   << " for absolute tolerance"
                   << "; found type="
                   << ode_fun.abs_tol_.expression_type()
                   << ". ";
        pass = false;
      }
      if (!ode_fun.max_num_steps_.expression_type().is_primitive()) {
        error_msgs << "tenth argument to integrate_ode_cvode"
                   << " must be type real or int"
                   << " for maximum number of steps"
                   << "; found type="
                   << ode_fun.max_num_steps_.expression_type()
                   << ". ";
        pass = false;
      }

      // test data-only variables do not have parameters (int locals OK)
      if (has_var(ode_fun.t0_, var_map)) {
        error_msgs << "third argument to integrate_ode_cvode (initial times)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.ts_, var_map)) {
        error_msgs << "fourth argument to integrate_ode_cvode"
                   << " (solution times) must be data only and not"
                   << " reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.x_, var_map)) {
        error_msgs << "fifth argument to integrate_ode_cvode (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.rel_tol_, var_map)) {
        error_msgs << "eight argument to integrate_ode_cvode (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.abs_tol_, var_map)) {
        error_msgs << "ninth argument to integrate_ode_cvode (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
      if (has_var(ode_fun.max_num_steps_, var_map)) {
        error_msgs << "tenth argument to integrate_ode_cvode (real data)"
                   << " must be data only and not reference parameters";
        pass = false;
      }
    }
    boost::phoenix::function<validate_integrate_ode_cvode>
    validate_integrate_ode_cvode_f;

    void set_fun_type_named::operator()(expression& fun_result, fun& fun,
                                        const var_origin& var_origin,
                                        bool& pass,
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
      fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                  arg_types,
                                                                  error_msgs);
      if (fun.type_ == ILL_FORMED_T) {
        pass = false;
        return;
      }

      if (has_rng_suffix(fun.name_)) {
        if (!( var_origin == derived_origin
               || var_origin == function_argument_origin_rng)) {
          error_msgs << "random number generators only allowed in"
                     << " generated quantities block or"
                     << " user-defined functions with names ending in _rng"
                     << "; found function=" << fun.name_
                     << " in block=";
          print_var_origin(error_msgs, var_origin);
          error_msgs << std::endl;
          pass = false;
          return;
        }
      }

      if (has_lp_suffix(fun.name_) || fun.name_ == "target") {
        // modified function_argument_origin to add _lp because
        // that's only viable context
        if (!(var_origin == transformed_parameter_origin
              || var_origin == function_argument_origin_lp
              || var_origin == void_function_argument_origin_lp
              || var_origin == local_origin)) {
          error_msgs << "Function target() or functions suffixed with _lp only"
                     << " allowed in transformed parameter block, model block"
                     << std::endl
                     << "or the body of a function with suffix _lp."
                     << std::endl
                     << "Found function = "
                     << (fun.name_ == "get_lp" ? "target or get_lp" : fun.name_)
                     << " in block = ";
          print_var_origin(error_msgs, var_origin);
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

      fun_result = fun;
      pass = true;
    }
    boost::phoenix::function<set_fun_type_named> set_fun_type_named_f;

    void exponentiation_expr::operator()(expression& expr1,
                                         const expression& expr2,
                                         const var_origin& var_origin,
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
        print_var_origin(error_msgs, var_origin);
        error_msgs << std::endl;
        pass = false;
        return;
      }
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      set_fun_type sft;
      fun f("pow", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("multiply", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
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
        sft(f, error_msgs);
        expr1 = expression(f);
        return;
      }
      if ((expr1.expression_type().type() == MATRIX_T
           || expr1.expression_type().type() == ROW_VECTOR_T)
          && expr2.expression_type().type() == MATRIX_T) {
        fun f("mdivide_right", args);
        sft(f, error_msgs);
        expr1 = expression(f);
        return;
      }
      fun f("divide", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("modulus", args);
      sft(f, error_msgs);
      expr1 = expression(f);
    }
    boost::phoenix::function<modulus_expr> modulus_f;

    void left_division_expr::operator()(expression& expr1, bool& pass,
                                        const expression& expr2,
                                        std::ostream& error_msgs) const {
      std::vector<expression> args;
      args.push_back(expr1);
      args.push_back(expr2);
      set_fun_type sft;
      if (expr1.expression_type().type() == MATRIX_T
          && (expr2.expression_type().type() == VECTOR_T
              || expr2.expression_type().type() == MATRIX_T)) {
        fun f("mdivide_left", args);
        sft(f, error_msgs);
        expr1 = expression(f);
        pass = true;
        return;
      }
      fun f("mdivide_left", args);  // set for alt args err msg
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("elt_multiply", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("elt_divide", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("minus", args);
      sft(f, error_msgs);
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
      set_fun_type sft;
      fun f("logical_negation", args);
      sft(f, error_msgs);
      expr_result = expression(f);
    }
    boost::phoenix::function<logical_negate_expr> logical_negate_expr_f;

    void transpose_expr::operator()(expression& expr, bool& pass,
                                    std::ostream& error_msgs) const {
      if (expr.expression_type().is_primitive())
        return;
      std::vector<expression> args;
      args.push_back(expr);
      set_fun_type sft;
      fun f("transpose", args);
      sft(f, error_msgs);
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
    bool data_only_expression::operator()(const array_literal& x) const {
      for (size_t i = 0; i < x.args_.size(); ++i)
        if (!boost::apply_visitor(*this, x.args_[i].expr_))
          return false;
      return true;
    }
    bool data_only_expression::operator()(const variable& x) const {
      var_origin origin = var_map_.get_origin(x.name_);
      bool is_data = (origin == data_origin)
        || (origin == transformed_data_origin)
        || (origin == local_origin);
      if (!is_data) {
        error_msgs_ << "non-data variables not allowed"
                    << " in dimension declarations."
                    << std::endl
                    << "     found variable=" << x.name_
                    << "; declared in block=";
        print_var_origin(error_msgs_, origin);
        error_msgs_ << std::endl;
      }
      return is_data;
    }
    bool data_only_expression::operator()(const integrate_ode& x) const {
      return boost::apply_visitor(*this, x.y0_.expr_)
        && boost::apply_visitor(*this, x.theta_.expr_);
    }
    bool data_only_expression::operator()(const integrate_ode_cvode& x) const {
      return boost::apply_visitor(*this, x.y0_.expr_)
        && boost::apply_visitor(*this, x.theta_.expr_);
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
      // Constant functions which can be used as identifiers
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


    void validate_int_expr::operator()(const expression& expr,
                                       bool& pass,
                                       std::stringstream& error_msgs) const {
      if (!expr.expression_type().is_primitive_int()) {
        error_msgs << "expression denoting integer required; found type="
                   << expr.expression_type() << std::endl;
        pass = false;
        return;
      }
      pass = true;
    }
    boost::phoenix::function<validate_int_expr> validate_int_expr_f;

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
                                            int var_origin, bool& pass,
                                            variable_map& var_map,
                                            std::stringstream& error_msgs)
      const {
      if (!expr.expression_type().is_primitive_int()) {
        error_msgs << "dimension declaration requires expression"
                   << " denoting integer; found type="
                   << expr.expression_type()
                   << std::endl;
        pass = false;
      } else if (var_origin != local_origin) {
        data_only_expression vis(error_msgs, var_map);
        bool only_data_dimensions = boost::apply_visitor(vis, expr.expr_);
        pass = only_data_dimensions;
      } else {
        // don't need to check data vs. parameter in dimensions for
        // local variable declarations
        pass = true;
      }
    }
    boost::phoenix::function<validate_int_data_expr> validate_int_data_expr_f;





    bool validate_double_expr::operator()(const expression& expr,
                                          std::stringstream& error_msgs) const {
      if (!expr.expression_type().is_primitive_double()
          && !expr.expression_type().is_primitive_int()) {
        error_msgs << "expression denoting real required; found type="
                   << expr.expression_type() << std::endl;
        return false;
      }
      return true;
    }
    boost::phoenix::function<validate_double_expr> validate_double_expr_f;

    void set_double_range_lower::operator()(range& range,
                                            const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      range.low_ = expr;
      validate_double_expr validator;
      pass = validator(expr, error_msgs);
    }
    boost::phoenix::function<set_double_range_lower> set_double_range_lower_f;

    void set_double_range_upper::operator()(range& range,
                                            const expression& expr,
                                            bool& pass,
                                            std::stringstream& error_msgs)
      const {
      range.high_ = expr;
      validate_double_expr validator;
      pass = validator(expr, error_msgs);
    }
    boost::phoenix::function<set_double_range_upper> set_double_range_upper_f;

    template <typename T>
    void add_var::operator()(var_decl& var_decl_result, const T& var_decl,
                             variable_map& vm, bool& pass, const var_origin& vo,
                             std::ostream& error_msgs) const {
      if (vm.exists(var_decl.name_)) {
        // variable already exists
        pass = false;
        error_msgs << "duplicate declaration of variable, name="
                   << var_decl.name_;

        error_msgs << "; attempt to redeclare as ";
        print_var_origin(error_msgs, vo);  // FIXME -- need original vo

        error_msgs << "; original declaration as ";
        print_var_origin(error_msgs, vm.get_origin(var_decl.name_));

        error_msgs << std::endl;
        var_decl_result = var_decl;
        return;
      }
      if ((vo == parameter_origin || vo == transformed_parameter_origin)
          && var_decl.base_type_ == INT_T) {
        pass = false;
        error_msgs << "integer parameters or transformed parameters"
                   << " are not allowed; "
                   << " found declared type int, parameter name="
                   << var_decl.name_
                   << std::endl;
        var_decl_result = var_decl;
        return;
      }
      pass = true;  // probably don't need to set true
      vm.add(var_decl.name_, var_decl, vo);
      var_decl_result = var_decl;
    }
    boost::phoenix::function<add_var> add_var_f;

    template void add_var::operator()(var_decl&, const int_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const double_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const vector_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const row_vector_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const matrix_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const simplex_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const unit_vector_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const ordered_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&,
                                      const positive_ordered_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&,
                                      const cholesky_factor_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const cholesky_corr_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const cov_matrix_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;
    template void add_var::operator()(var_decl&, const corr_matrix_var_decl&,
                                      variable_map&, bool&, const var_origin&,
                                      std::ostream&) const;


  }
}

#endif
