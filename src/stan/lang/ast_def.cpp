#ifndef STAN_LANG_AST_DEF_CPP
#define STAN_LANG_AST_DEF_CPP

#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <stan/lang/ast.hpp>

#include <cstddef>
#include <limits>
#include <climits>
#include <istream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stan {

  namespace lang {

    std::ostream& write_base_expr_type(std::ostream& o, base_expr_type type) {
      switch (type) {
      case INT_T :
        o << "int";
        break;
      case DOUBLE_T :
        o << "real";
        break;
      case VECTOR_T :
        o << "vector";
        break;
      case ROW_VECTOR_T :
        o << "row vector";
        break;
      case MATRIX_T :
        o << "matrix";
        break;
      case ILL_FORMED_T :
        o << "ill formed";
        break;
      case VOID_T :
        o << "void";
        break;
      default:
        o << "UNKNOWN";
      }
      return o;
    }

    // expr_type ctors and methods
    expr_type::expr_type()
      : base_type_(ILL_FORMED_T),
        num_dims_(0) {
    }
    expr_type::expr_type(const base_expr_type base_type)
      : base_type_(base_type),
        num_dims_(0) {
    }
    expr_type::expr_type(const base_expr_type base_type,
                         size_t num_dims)
      : base_type_(base_type),
        num_dims_(num_dims) {
    }
    bool expr_type::operator==(const expr_type& et) const {
      return base_type_ == et.base_type_
        && num_dims_ == et.num_dims_;
    }
    bool expr_type::operator!=(const expr_type& et) const {
        return !(*this == et);
    }
    bool expr_type::operator<(const expr_type& et) const {
      return (base_type_ < et.base_type_)
        || (base_type_ == et.base_type_
            && num_dims_ < et.num_dims_);
    }
    bool expr_type::operator<=(const expr_type& et) const {
      return (base_type_ < et.base_type_)
        || (base_type_ == et.base_type_
            && num_dims_ <= et.num_dims_);
    }
    bool expr_type::operator>(const expr_type& et) const {
      return (base_type_ > et.base_type_)
        || (base_type_ == et.base_type_
            && num_dims_ > et.num_dims_);
    }
    bool expr_type::operator>=(const expr_type& et) const {
      return (base_type_ > et.base_type_)
        || (base_type_ == et.base_type_
            && num_dims_ >= et.num_dims_);
    }
    bool expr_type::is_primitive() const {
      return is_primitive_int()
        || is_primitive_double();
    }
    bool expr_type::is_primitive_int() const {
      return base_type_ == INT_T
        && num_dims_ == 0U;
    }
    bool expr_type::is_primitive_double() const {
      return base_type_ == DOUBLE_T
        && num_dims_ == 0U;
    }
    bool expr_type::is_ill_formed() const {
      return base_type_ == ILL_FORMED_T;
    }
    bool expr_type::is_void() const {
      return base_type_ == VOID_T;
    }
    base_expr_type expr_type::type() const {
      return base_type_;
    }
    size_t expr_type::num_dims() const {
      return num_dims_;
    }

    // output matches unsized types used to declare functions
    std::ostream& operator<<(std::ostream& o, const expr_type& et) {
      write_base_expr_type(o, et.type());
      if (et.num_dims() > 0) {
        o << '[';
        for (size_t i = 1; i < et.num_dims(); ++i)
          o << ",";
        o << ']';
      }
      return o;
    }

    expr_type promote_primitive(const expr_type& et) {
      if (!et.is_primitive())
        return expr_type();
      return et;
    }

    expr_type promote_primitive(const expr_type& et1,
                                const expr_type& et2) {
      if (!et1.is_primitive() || !et2.is_primitive())
        return expr_type();
      return et1.type() == DOUBLE_T ? et1 : et2;
    }

    void function_signatures::reset_sigs() {
      if (sigs_ == 0) return;
      delete sigs_;
      sigs_ = 0;
    }
    function_signatures& function_signatures::instance() {
      // FIXME:  for threaded models, requires double-check lock
      if (!sigs_)
        sigs_ = new function_signatures;
      return *sigs_;
    }
    void
    function_signatures::set_user_defined(const
                                          std::pair<std::string,
                                                    function_signature_t>&
                                          name_sig) {
      user_defined_set_.insert(name_sig);
    }
    bool
    function_signatures::is_user_defined(const std::pair<std::string,
                                                         function_signature_t>&
                                              name_sig) {
      return user_defined_set_.find(name_sig) != user_defined_set_.end();
    }
    bool function_signatures::is_defined(const std::string& name,
                                         const function_signature_t& sig) {
      if (sigs_map_.find(name) == sigs_map_.end())
        return false;
      const std::vector<function_signature_t> sigs = sigs_map_[name];
      for (size_t i = 0; i < sigs.size(); ++i)
        if (sig.second  == sigs[i].second)
          return true;
      return false;
    }

    void function_signatures::add(const std::string& name,
                                   const expr_type& result_type,
                                   const std::vector<expr_type>& arg_types) {
      sigs_map_[name].push_back(function_signature_t(result_type, arg_types));
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type) {
      std::vector<expr_type> arg_types;
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2,
                                  const expr_type& arg_type3) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      arg_types.push_back(arg_type3);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2,
                                  const expr_type& arg_type3,
                                  const expr_type& arg_type4) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      arg_types.push_back(arg_type3);
      arg_types.push_back(arg_type4);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2,
                                  const expr_type& arg_type3,
                                  const expr_type& arg_type4,
                                  const expr_type& arg_type5) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      arg_types.push_back(arg_type3);
      arg_types.push_back(arg_type4);
      arg_types.push_back(arg_type5);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2,
                                  const expr_type& arg_type3,
                                  const expr_type& arg_type4,
                                  const expr_type& arg_type5,
                                  const expr_type& arg_type6) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      arg_types.push_back(arg_type3);
      arg_types.push_back(arg_type4);
      arg_types.push_back(arg_type5);
      arg_types.push_back(arg_type6);
      add(name, result_type, arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2,
                                  const expr_type& arg_type3,
                                  const expr_type& arg_type4,
                                  const expr_type& arg_type5,
                                  const expr_type& arg_type6,
                                  const expr_type& arg_type7) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      arg_types.push_back(arg_type3);
      arg_types.push_back(arg_type4);
      arg_types.push_back(arg_type5);
      arg_types.push_back(arg_type6);
      arg_types.push_back(arg_type7);
      add(name, result_type, arg_types);
    }
    void function_signatures::add_nullary(const::std::string& name) {
      add(name, DOUBLE_T);
    }
    void function_signatures::add_unary(const::std::string& name) {
      add(name, DOUBLE_T, DOUBLE_T);
    }
    void function_signatures::add_unary_vectorized(const::std::string& 
    name) {
      for (size_t i = 0; i < 8; ++i) {
        add(name, expr_type(DOUBLE_T, i), expr_type(INT_T, i));
        add(name, expr_type(DOUBLE_T, i), expr_type(DOUBLE_T, i));
        add(name, expr_type(VECTOR_T, i), expr_type(VECTOR_T, i));
        add(name, expr_type(ROW_VECTOR_T, i), expr_type(ROW_VECTOR_T, i));
        add(name, expr_type(MATRIX_T, i), expr_type(MATRIX_T, i));
      }
    }
    void function_signatures::add_binary(const::std::string& name) {
      add(name, DOUBLE_T, DOUBLE_T, DOUBLE_T);
    }
    void function_signatures::add_ternary(const::std::string& name) {
      add(name, DOUBLE_T, DOUBLE_T, DOUBLE_T, DOUBLE_T);
    }
    void function_signatures::add_quaternary(const::std::string& name) {
      add(name, DOUBLE_T, DOUBLE_T, DOUBLE_T, DOUBLE_T, DOUBLE_T);
    }
    int function_signatures::num_promotions(
                            const std::vector<expr_type>& call_args,
                            const std::vector<expr_type>& sig_args) {
      if (call_args.size() != sig_args.size()) {
        return -1;  // failure
      }
      int num_promotions = 0;
      for (size_t i = 0; i < call_args.size(); ++i) {
        if (call_args[i] == sig_args[i]) {
          continue;
        } else if (call_args[i].is_primitive_int()
                   && sig_args[i].is_primitive_double()) {
          ++num_promotions;
        } else {
          return -1;  // failed match
        }
      }
      return num_promotions;
    }
    int function_signatures::get_signature_matches(const std::string& name,
                              const std::vector<expr_type>& args,
                              function_signature_t& signature) {
      if (!has_key(name)) return 0;
      std::vector<function_signature_t> signatures = sigs_map_[name];
      size_t min_promotions = std::numeric_limits<size_t>::max();
      size_t num_matches = 0;
      for (size_t i = 0; i < signatures.size(); ++i) {
        signature = signatures[i];
        int promotions = num_promotions(args, signature.second);
        if (promotions < 0) continue;  // no match
        size_t promotions_ui = static_cast<size_t>(promotions);
        if (promotions_ui < min_promotions) {
          min_promotions = promotions_ui;
          num_matches = 1;
        } else if (promotions_ui == min_promotions) {
          ++num_matches;
        }
      }
      return num_matches;
    }

    bool is_binary_operator(const std::string& name) {
      return name == "add"
        || name == "subtract"
        || name == "multiply"
        || name == "divide"
        || name == "modulus"
        || name == "mdivide_left"
        || name == "mdivide_right"
        || name == "elt_multiply"
        || name == "elt_divide";
    }

    bool is_unary_operator(const std::string& name) {
      return name == "minus"
        || name == "logical_negation";
    }

    bool is_unary_postfix_operator(const std::string& name) {
      return name == "transpose";
    }

    bool is_operator(const std::string& name) {
      return is_binary_operator(name)
        || is_unary_operator(name)
        || is_unary_postfix_operator(name);
    }

    std::string fun_name_to_operator(const std::string& name) {
      // binary infix (pow handled by parser)
      if (name == "add") return "+";
      if (name == "subtract") return "-";
      if (name == "multiply") return "*";
      if (name == "divide") return "/";
      if (name == "modulus") return "%";
      if (name == "mdivide_left") return "\\";
      if (name == "mdivide_right") return "/";
      if (name == "elt_multiply") return ".*";
      if (name == "elt_divide") return "./";

      // unary prefix (+ handled by parser)
      if (name == "minus") return "-";
      if (name == "logical_negation") return "!";

      // unary suffix
      if (name == "transpose") return "'";

      // none of the above
      return "ERROR";
    }

    void print_signature(const std::string& name,
                         const std::vector<expr_type>& arg_types,
                         bool sampling_error_style,
                         std::ostream& msgs) {
      static size_t OP_SIZE = std::string("operator").size();
      msgs << "  ";
      if (name.size() > OP_SIZE && name.substr(0, OP_SIZE) == "operator") {
        std::string operator_name = name.substr(OP_SIZE);
        if (arg_types.size() == 2) {
          msgs << arg_types[0] << " " << operator_name << " " << arg_types[1]
               << std::endl;
          return;
        } else if (arg_types.size() == 1) {
          if (operator_name == "'")  // exception for postfix
            msgs << arg_types[0] << operator_name << std::endl;
          else
            msgs << operator_name << arg_types[0] << std::endl;
          return;
        } else {
          // should not be reachable due to operator grammar
          // continue on purpose to get more info to user if this happens
          msgs << "Operators must have 1 or 2 arguments." << std::endl;
        }
      }
      if (sampling_error_style && arg_types.size() > 0)
        msgs << arg_types[0] << " ~ ";
      msgs << name << "(";
      size_t start = sampling_error_style ? 1 : 0;
      for (size_t j = start; j < arg_types.size(); ++j) {
        if (j > start) msgs << ", ";
        msgs << arg_types[j];
      }
      msgs << ")" << std::endl;
    }

    expr_type function_signatures::get_result_type(const std::string& name,
                                           const std::vector<expr_type>& args,
                                           std::ostream& error_msgs,
                                           bool sampling_error_style) {
      std::vector<function_signature_t> signatures = sigs_map_[name];
      size_t match_index = 0;
      size_t min_promotions = std::numeric_limits<size_t>::max();
      size_t num_matches = 0;

      std::string display_name;
      if (is_operator(name)) {
        display_name = "operator" + fun_name_to_operator(name);
      } else if (sampling_error_style && ends_with("_log", name)) {
        display_name = name.substr(0, name.size() - 4);
      } else if (sampling_error_style
                 && (ends_with("_lpdf", name) || ends_with("_lcdf", name))) {
        display_name = name.substr(0, name.size() - 5);
      } else {
        display_name = name;
      }

      for (size_t i = 0; i < signatures.size(); ++i) {
        int promotions = num_promotions(args, signatures[i].second);
        if (promotions < 0) continue;  // no match
        size_t promotions_ui = static_cast<size_t>(promotions);
        if (promotions_ui < min_promotions) {
          min_promotions = promotions_ui;
          match_index = i;
          num_matches = 1;
        } else if (promotions_ui == min_promotions) {
          ++num_matches;
        }
      }

      if (num_matches == 1)
        return signatures[match_index].first;

      // all returns after here are for ill-typed input

      if (num_matches == 0) {
        error_msgs << "No matches for: "
                   << std::endl << std::endl;
      } else {
        error_msgs << "Ambiguous: "
                   << num_matches << " matches with "
                   << min_promotions << " integer promotions for: "
                   << std::endl;
      }
      print_signature(display_name, args, sampling_error_style, error_msgs);

      if (signatures.size() == 0) {
        error_msgs << std::endl
                   << (sampling_error_style ? "Distribution " : "Function ")
                   << display_name << " not found.";
        if (sampling_error_style)
          error_msgs << " Require function with _lpdf or _lpmf or _log suffix";
        error_msgs << std::endl;
      } else {
        error_msgs << std::endl
                   << "Available argument signatures for "
                   << display_name << ":" << std::endl << std::endl;

        for (size_t i = 0; i < signatures.size(); ++i) {
          print_signature(display_name, signatures[i].second,
                          sampling_error_style, error_msgs);
        }
        error_msgs << std::endl;
      }
      return expr_type();  // ill-formed dummy
    }

    function_signatures::function_signatures() {
#include <stan/lang/function_signatures.h>  // NOLINT
    }

    bool function_signatures::has_user_defined_key(const std::string& key)
      const {
      using std::pair;
      using std::set;
      using std::string;
      for (set<pair<string, function_signature_t> >::const_iterator
             it = user_defined_set_.begin();
           it != user_defined_set_.end();
           ++it) {
        if (it->first == key)
          return true;
      }
      return false;
    }

    std::set<std::string> function_signatures::key_set() const {
      using std::map;
      using std::set;
      using std::string;
      using std::vector;
      set<string> result;
      for (map<string, vector<function_signature_t> >::const_iterator
             it = sigs_map_.begin();
           it != sigs_map_.end();
           ++it)
        result.insert(it->first);
      return result;
    }

    bool function_signatures::has_key(const std::string& key) const {
      return sigs_map_.find(key) != sigs_map_.end();
    }

    function_signatures* function_signatures::sigs_ = 0;

    arg_decl::arg_decl() { }
    arg_decl::arg_decl(const expr_type& arg_type,
                       const std::string& name)
      : arg_type_(arg_type),
        name_(name) {
    }
    base_var_decl arg_decl::base_variable_declaration() {
      std::vector<expression> dims;
      for (size_t i = 0; i < arg_type_.num_dims_; ++i)
        dims.push_back(expression(int_literal(0)));  // dummy value 0
      return base_var_decl(name_, dims, arg_type_.base_type_);
    }

    function_decl_def::function_decl_def() { }
    function_decl_def::function_decl_def(const expr_type& return_type,
                                         const std::string& name,
                                         const std::vector<arg_decl>& arg_decls,
                                         const statement& body)

      : return_type_(return_type),
        name_(name),
        arg_decls_(arg_decls),
        body_(body) {
    }

    function_decl_defs::function_decl_defs() { }
    function_decl_defs::function_decl_defs(const std::vector<function_decl_def>&
                                           decl_defs)
      : decl_defs_(decl_defs) {
    }

    returns_type_vis::returns_type_vis(const expr_type& return_type,
                                       std::ostream& error_msgs)
      : return_type_(return_type),
        error_msgs_(error_msgs) {
    }
    bool returns_type_vis::operator()(const nil& st) const {
      error_msgs_ << "Expecting return, found nil statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const assignment& st) const {
      error_msgs_ << "Expecting return, found assignment statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const assgn& st) const {
      error_msgs_ << "Expecting return, found assignment statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const sample& st) const {
      error_msgs_ << "Expecting return, found sampling statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const
                                      increment_log_prob_statement& t) const {
      error_msgs_ << "Expecting return, found increment_log_prob statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const expression& st) const  {
      error_msgs_ << "Expecting return, found increment_log_prob statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const print_statement& st) const  {
      error_msgs_ << "Expecting return, found print statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const reject_statement& st) const  {
      error_msgs_ << "Expecting return, found reject statement."
                 << std::endl;
      return false;
    }
    bool returns_type_vis::operator()(const no_op_statement& st) const  {
      error_msgs_ << "Expecting return, found no_op statement."
                 << std::endl;
      return false;
    }
    // recursive cases
    bool returns_type_vis::operator()(const statements& st) const  {
      // last statement in sequence must return type
      if (st.statements_.size() == 0) {
        error_msgs_ << ("Expecting return, found"
                        " statement sequence with empty body.")
                    << std::endl;
        return false;
      }
      return returns_type(return_type_, st.statements_.back(), error_msgs_);
    }
    bool returns_type_vis::operator()(const for_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.statement_, error_msgs_);
    }
    bool returns_type_vis::operator()(const while_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.body_, error_msgs_);
    }
    bool returns_type_vis::operator()(const break_continue_statement& st)
      const  {
      // break/continue OK only as end of nested loop in void return
      bool pass = (return_type_ == VOID_T);
      if (!pass)
        error_msgs_ << "statement " << st.generate_
                    << " does not match return type";
      return pass;
    }
    bool returns_type_vis::operator()(const
                                      conditional_statement& st) const  {
      // all condition bodies must end in appropriate return
      if (st.bodies_.size() != (st.conditions_.size() + 1)) {
        error_msgs_ << ("Expecting return, found conditional"
                        " without final else.")
                    << std::endl;
        return false;
      }
      for (size_t i = 0; i < st.bodies_.size(); ++i)
        if (!returns_type(return_type_, st.bodies_[i], error_msgs_))
          return false;
      return true;
    }
    bool returns_type_vis::operator()(const return_statement& st) const  {
      // return checked for type
      return return_type_ == VOID_T
        || is_assignable(return_type_, st.return_value_.expression_type(),
                         "Returned expression does not match return type",
                         error_msgs_);
    }

    bool returns_type(const expr_type& return_type,
                      const statement& statement,
                      std::ostream& error_msgs) {
      if (return_type == VOID_T)
        return true;
      returns_type_vis vis(return_type, error_msgs);
      return boost::apply_visitor(vis, statement.statement_);
    }



    statements::statements() {  }
    statements::statements(const std::vector<var_decl>& local_decl,
                           const std::vector<statement>& stmts)
      : local_decl_(local_decl),
        statements_(stmts) {
    }

    expr_type expression_type_vis::operator()(const nil& /*e*/) const {
      return expr_type();
    }
    expr_type expression_type_vis::operator()(const int_literal& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const double_literal& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const array_literal& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const variable& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const integrate_ode& e) const {
      return expr_type(DOUBLE_T, 2);
    }
    expr_type
    expression_type_vis::operator()(const integrate_ode_control& e) const {
      return expr_type(DOUBLE_T, 2);
    }
    expr_type expression_type_vis::operator()(const fun& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const index_op& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const index_op_sliced& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const conditional_op& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const binary_op& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const unary_op& e) const {
      return e.type_;
    }

    expression::expression()
      : expr_(nil()) {
    }
    expression::expression(const expression& e)
      : expr_(e.expr_) {
    }
    expr_type expression::expression_type() const {
      expression_type_vis vis;
      return boost::apply_visitor(vis, expr_);
    }

    expression::expression(const expression_t& expr) : expr_(expr) { }
    expression::expression(const nil& expr) : expr_(expr) { }
    expression::expression(const int_literal& expr) : expr_(expr) { }
    expression::expression(const double_literal& expr) : expr_(expr) { }
    expression::expression(const array_literal& expr) : expr_(expr) { }
    expression::expression(const variable& expr) : expr_(expr) { }
    expression::expression(const integrate_ode& expr) : expr_(expr) { }
    expression::expression(const integrate_ode_control& expr) : expr_(expr) { }
    expression::expression(const fun& expr) : expr_(expr) { }
    expression::expression(const index_op& expr) : expr_(expr) { }
    expression::expression(const index_op_sliced& expr) : expr_(expr) { }
    expression::expression(const conditional_op& expr) : expr_(expr) { }
    expression::expression(const binary_op& expr) : expr_(expr) { }
    expression::expression(const unary_op& expr) : expr_(expr) { }

    int expression::total_dims() const {
      int sum = expression_type().num_dims_;
      if (expression_type().type() == VECTOR_T)
        ++sum;
      if (expression_type().type() == ROW_VECTOR_T)
        ++sum;
      if (expression_type().type() == MATRIX_T)
        sum += 2;
      return sum;
    }

    printable::printable() : printable_("") { }
    printable::printable(const expression& expr) : printable_(expr) { }
    printable::printable(const std::string& msg) : printable_(msg) { }
    printable::printable(const printable_t& printable)
      : printable_(printable) { }
    printable::printable(const printable& printable)
      : printable_(printable.printable_) { }

    contains_var::contains_var(const variable_map& var_map)
      : var_map_(var_map) {
    }
    bool contains_var::operator()(const nil& e) const {
      return false;
    }
    bool contains_var::operator()(const int_literal& e) const {
      return false;
    }
    bool contains_var::operator()(const double_literal& e) const {
      return false;
    }
    bool contains_var::operator()(const array_literal& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_var::operator()(const variable& e) const {
      var_origin vo = var_map_.get_origin(e.name_);
      return vo == parameter_origin
        || vo == transformed_parameter_origin
        || (vo == local_origin && e.type_.base_type_ != INT_T);
    }
    bool contains_var::operator()(const fun& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_var::operator()(const integrate_ode& e) const {
      // only init state and params may contain vars
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }
    bool contains_var::operator()(const integrate_ode_control& e) const {
      // only init state and params may contain vars
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }
    bool contains_var::operator()(const index_op& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }
    bool contains_var::operator()(const index_op_sliced& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }
    bool contains_var::operator()(const conditional_op& e) const {
      return boost::apply_visitor(*this, e.cond_.expr_)
        || boost::apply_visitor(*this, e.true_val_.expr_)
        || boost::apply_visitor(*this, e.false_val_.expr_);
    }
    bool contains_var::operator()(const binary_op& e) const {
      return boost::apply_visitor(*this, e.left.expr_)
        || boost::apply_visitor(*this, e.right.expr_);
    }
    bool contains_var::operator()(const unary_op& e) const {
        return boost::apply_visitor(*this, e.subject.expr_);
    }

    bool is_linear_function(const std::string& name) {
      return name == "add"
        || name == "block"
        || name == "append_col"
        || name == "col"
        || name == "cols"
        || name == "diagonal"
        || name == "head"
        || name == "minus"
        || name == "negative_infinity"
        || name == "not_a_number"
        || name == "append_row"
        || name == "rep_matrix"
        || name == "rep_row_vector"
        || name == "rep_vector"
        || name == "row"
        || name == "rows"
        || name == "positive_infinity"
        || name == "segment"
        || name == "subtract"
        || name == "sum"
        || name == "tail"
        || name == "to_vector"
        || name == "to_row_vector"
        || name == "to_matrix"
        || name == "to_array_1d"
        || name == "to_array_2d"
        || name == "transpose";
    }

    bool has_var(const expression& e,
                 const variable_map& var_map) {
      contains_var vis(var_map);
      return boost::apply_visitor(vis, e.expr_);
    }

    contains_nonparam_var::contains_nonparam_var(const variable_map& var_map)
      : var_map_(var_map) {
    }
    bool contains_nonparam_var::operator()(const nil& e) const {
      return false;
    }
    bool contains_nonparam_var::operator()(const int_literal& e) const {
      return false;
    }
    bool contains_nonparam_var::operator()(const double_literal& e) const {
      return false;
    }
    bool contains_nonparam_var::operator()(const array_literal& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_nonparam_var::operator()(const variable& e) const {
      var_origin vo = var_map_.get_origin(e.name_);
      return (vo == transformed_parameter_origin
              || vo == local_origin);
    }
    bool contains_nonparam_var::operator()(const integrate_ode& e) const {
      // if any vars, return true because integration will be nonlinear
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }
    bool contains_nonparam_var::operator()(const integrate_ode_control& e)
      const {
      // if any vars, return true because integration will be nonlinear
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }
    bool contains_nonparam_var::operator()(const fun& e) const {
      // any function applied to non-linearly transformed var
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      // non-linear function applied to var
      if (!is_linear_function(e.name_)) {
        for (size_t i = 0; i < e.args_.size(); ++i)
          if (has_var(e.args_[i], var_map_))
            return true;
      }
      return false;
    }
    bool contains_nonparam_var::operator()(const index_op& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }
    bool contains_nonparam_var::operator()(const index_op_sliced& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }

    bool contains_nonparam_var::operator()(const conditional_op& e) const {
      if (has_non_param_var(e.cond_, var_map_)
          || has_non_param_var(e.true_val_, var_map_)
          || has_non_param_var(e.false_val_, var_map_))
        return true;
      return false;
    }

    bool contains_nonparam_var::operator()(const binary_op& e) const {
      if (e.op == "||"
          || e.op == "&&"
          || e.op == "=="
          || e.op == "!="
          || e.op == "<"
          || e.op == "<="
          || e.op == ">"
          || e.op == ">=")
        return true;
      if (has_non_param_var(e.left, var_map_)
          || has_non_param_var(e.right, var_map_))
        return true;
      if (e.op == "*" || e.op == "/")
        return has_var(e.left, var_map_) && has_var(e.right, var_map_);
      return false;
    }
    bool contains_nonparam_var::operator()(const unary_op& e) const {
      // only negation, which is linear, so recurse
      return has_non_param_var(e.subject, var_map_);
    }

    bool has_non_param_var(const expression& e,
                           const variable_map& var_map) {
      contains_nonparam_var vis(var_map);
      return boost::apply_visitor(vis, e.expr_);
    }

    bool is_nil_op::operator()(const nil& /*x*/) const { return true; }
    bool is_nil_op::operator()(const int_literal& /*x*/) const { return false; }
    bool is_nil_op::operator()(const double_literal& /* x */) const {
      return false;
    }
    bool is_nil_op::operator()(const array_literal& /* x */)
      const { return false; }
    bool is_nil_op::operator()(const variable& /* x */) const { return false; }
    bool is_nil_op::operator()(const integrate_ode& /* x */) const {
      return false;
    }
    bool is_nil_op::operator()(const integrate_ode_control& /* x */) const {
      return false;
    }
    bool is_nil_op::operator()(const fun& /* x */) const { return false; }
    bool is_nil_op::operator()(const index_op& /* x */) const { return false; }
    bool is_nil_op::operator()(const index_op_sliced& /* x */) const {
      return false;
    }
    bool is_nil_op::operator()(const conditional_op& /* x */) const {
      return false; }
    bool is_nil_op::operator()(const binary_op& /* x */) const { return false; }
    bool is_nil_op::operator()(const unary_op& /* x */) const { return false; }

    bool is_nil(const expression& e) {
      is_nil_op ino;
      return boost::apply_visitor(ino, e.expr_);
    }

    variable_dims::variable_dims() { }  // req for FUSION_ADAPT
    variable_dims::variable_dims(std::string const& name,
                                 std::vector<expression> const& dims)
      : name_(name),
        dims_(dims) {
    }


    int_literal::int_literal()
      : type_(INT_T) {
    }
    int_literal::int_literal(int val)
      : val_(val),
        type_(INT_T) {
    }
    int_literal::int_literal(const int_literal& il)
      : val_(il.val_),
        type_(il.type_) {
    }
    int_literal& int_literal::operator=(const int_literal& il) {
      val_ = il.val_;
      type_ = il.type_;
      return *this;
    }


    double_literal::double_literal()
      : type_(DOUBLE_T, 0U) {
    }
    double_literal::double_literal(double val)
      : val_(val),
        type_(DOUBLE_T, 0U) {
    }
    double_literal& double_literal::operator=(const double_literal& dl) {
      val_ = dl.val_;
      type_ = dl.type_;
      return *this;
    }


    array_literal::array_literal()
      : args_(),
        type_(DOUBLE_T, 1U) {
    }
    array_literal::array_literal(const std::vector<expression>& args)
      : args_(args),
        type_() {  // ill-formed w/o help
    }
    array_literal& array_literal::operator=(const array_literal& al) {
      args_ = al.args_;
      type_ = al.type_;
      return *this;
    }

    variable::variable() { }
    variable::variable(std::string name) : name_(name) { }
    void variable::set_type(const base_expr_type& base_type,
                            size_t num_dims) {
      type_ = expr_type(base_type, num_dims);
    }

    integrate_ode::integrate_ode() { }
    integrate_ode::integrate_ode(const std::string& integration_function_name,
                                 const std::string& system_function_name,
                                 const expression& y0,
                                 const expression& t0,
                                 const expression& ts,
                                 const expression& theta,
                                 const expression& x,
                                 const expression& x_int)
      : integration_function_name_(integration_function_name),
        system_function_name_(system_function_name),
        y0_(y0),
        t0_(t0),
        ts_(ts),
        theta_(theta),
        x_(x),
        x_int_(x_int) {
    }

    integrate_ode_control::integrate_ode_control() { }
    integrate_ode_control::integrate_ode_control(
                                   const std::string& integration_function_name,
                                   const std::string& system_function_name,
                                   const expression& y0,
                                   const expression& t0,
                                   const expression& ts,
                                   const expression& theta,
                                   const expression& x,
                                   const expression& x_int,
                                   const expression& rel_tol,
                                   const expression& abs_tol,
                                   const expression& max_num_steps)
      : integration_function_name_(integration_function_name),
        system_function_name_(system_function_name),
        y0_(y0),
        t0_(t0),
        ts_(ts),
        theta_(theta),
        x_(x),
        x_int_(x_int),
        rel_tol_(rel_tol),
        abs_tol_(abs_tol),
        max_num_steps_(max_num_steps) {
    }

    fun::fun() { }
    fun::fun(std::string const& name,
             std::vector<expression> const& args)
      : name_(name),
        args_(args) {
    }

    size_t total_dims(const std::vector<std::vector<expression> >& dimss) {
      size_t total = 0U;
      for (size_t i = 0; i < dimss.size(); ++i)
        total += dimss[i].size();
      return total;
    }

    expr_type infer_type_indexing(const base_expr_type& expr_base_type,
                                  size_t num_expr_dims,
                                  size_t num_index_dims) {
      if (num_index_dims <= num_expr_dims)
        return expr_type(expr_base_type, num_expr_dims - num_index_dims);
      if (num_index_dims == (num_expr_dims + 1)) {
        if (expr_base_type == VECTOR_T || expr_base_type == ROW_VECTOR_T)
          return expr_type(DOUBLE_T, 0U);
        if (expr_base_type == MATRIX_T)
          return expr_type(ROW_VECTOR_T, 0U);
      }
      if (num_index_dims == (num_expr_dims + 2))
        if (expr_base_type == MATRIX_T)
          return expr_type(DOUBLE_T, 0U);

      // error condition, result expr_type has is_ill_formed() = true
      return expr_type();
    }

    expr_type infer_type_indexing(const expression& expr,
                                  size_t num_index_dims) {
      return infer_type_indexing(expr.expression_type().base_type_,
                                 expr.expression_type().num_dims(),
                                 num_index_dims);
    }


    index_op::index_op() { }
    index_op::index_op(const expression& expr,
                       const std::vector<std::vector<expression> >& dimss)
      : expr_(expr),
        dimss_(dimss) {
      infer_type();
    }
    void index_op::infer_type() {
      type_ = infer_type_indexing(expr_, total_dims(dimss_));
    }

    index_op_sliced::index_op_sliced() { }
    index_op_sliced::index_op_sliced(const expression& expr,
                                     const std::vector<idx>& idxs)
      : expr_(expr), idxs_(idxs), type_(indexed_type(expr_, idxs_)) { }
    void index_op_sliced::infer_type() {
      type_ = indexed_type(expr_, idxs_);
    }

    conditional_op::conditional_op() { }
    conditional_op::conditional_op(const expression& cond,
                         const expression& true_val,
                         const expression& false_val)
      : cond_(cond),
        true_val_(true_val),
        false_val_(false_val),
        type_(promote_primitive(true_val.expression_type(),
                                false_val.expression_type())) {
    }

    binary_op::binary_op() { }
    binary_op::binary_op(const expression& left,
                         const std::string& op,
                         const expression& right)
      : op(op),
        left(left),
        right(right),
        type_(promote_primitive(left.expression_type(),
                                  right.expression_type())) {
    }

    unary_op::unary_op(char op,
                       expression const& subject)
      : op(op),
        subject(subject),
        type_(promote_primitive(subject.expression_type())) {
    }


    range::range() { }
    range::range(expression const& low,
                 expression const& high)
      : low_(low),
        high_(high) {
    }
    bool range::has_low() const {
      return !is_nil(low_.expr_);
    }
    bool range::has_high() const {
      return !is_nil(high_.expr_);
    }

    uni_idx::uni_idx() { }
    uni_idx::uni_idx(const expression& idx) : idx_(idx) { }

    multi_idx::multi_idx() { }
    multi_idx::multi_idx(const expression& idxs) : idxs_(idxs) { }

    omni_idx::omni_idx() { }

    lb_idx::lb_idx() { }
    lb_idx::lb_idx(const expression& lb) : lb_(lb) { }

    ub_idx::ub_idx() { }
    ub_idx::ub_idx(const expression& ub) : ub_(ub) { }

    lub_idx::lub_idx() { }
    lub_idx::lub_idx(const expression& lb, const expression& ub)
      : lb_(lb), ub_(ub) {
    }

    idx::idx() { }

    idx::idx(const uni_idx& i) : idx_(i) { }
    idx::idx(const multi_idx& i) : idx_(i) { }
    idx::idx(const omni_idx& i) : idx_(i) { }
    idx::idx(const lb_idx& i) : idx_(i) { }
    idx::idx(const ub_idx& i) : idx_(i) { }
    idx::idx(const lub_idx& i) : idx_(i) { }


    is_multi_index_vis::is_multi_index_vis() { }
    bool is_multi_index_vis::operator()(const uni_idx& i) const {
      return false;
    }
    bool is_multi_index_vis::operator()(const multi_idx& i) const {
      return true;
    }
    bool is_multi_index_vis::operator()(const omni_idx& i) const {
      return true;
    }
    bool is_multi_index_vis::operator()(const lb_idx& i) const {
      return true;
    }
    bool is_multi_index_vis::operator()(const ub_idx& i) const {
      return true;
    }
    bool is_multi_index_vis::operator()(const lub_idx& i) const {
      return true;
    }

    bool is_multi_index(const idx& idx) {
      is_multi_index_vis v;
      return boost::apply_visitor(v, idx.idx_);
    }

    bool is_data_origin(const var_origin& vo) {
      return vo == data_origin || vo == transformed_data_origin;
    }

    void print_var_origin(std::ostream& o, const var_origin& vo) {
      if (vo == model_name_origin)
        o << "model name";
      else if (vo == data_origin)
        o << "data";
      else if (vo == transformed_data_origin)
        o << "transformed data";
      else if (vo == parameter_origin)
        o << "parameter";
      else if (vo == transformed_parameter_origin)
        o << "transformed parameter";
      else if (vo == derived_origin)
        o << "generated quantities";
      else if (vo == local_origin)
        o << "local";
      else if (vo == function_argument_origin)
        o << "function argument";
      else if (vo == function_argument_origin_lp)
        o << "function argument '_lp' suffixed";
      else if (vo == function_argument_origin_rng)
        o << "function argument '_rng' suffixed";
      else if (vo == void_function_argument_origin)
        o << "void function argument";
      else if (vo == void_function_argument_origin_lp)
        o << "void function argument '_lp' suffixed";
      else if (vo == void_function_argument_origin_rng)
        o << "void function argument '_rng' suffixed";
      else
        o << "UNKNOWN ORIGIN=" << vo;
    }


    base_var_decl::base_var_decl() { }
    base_var_decl::base_var_decl(const base_expr_type& base_type)
      : base_type_(base_type) {
    }
    base_var_decl::base_var_decl(const std::string& name,
                                 const std::vector<expression>& dims,
                                 const base_expr_type& base_type)
      : name_(name), dims_(dims), base_type_(base_type) {  }

    bool variable_map::exists(const std::string& name) const {
      return map_.find(name) != map_.end();
    }
    base_var_decl variable_map::get(const std::string& name) const {
      if (!exists(name))
        throw std::invalid_argument("variable does not exist");
      return map_.find(name)->second.first;
    }
    base_expr_type variable_map::get_base_type(const std::string& name) const {
      return get(name).base_type_;
    }
    size_t variable_map::get_num_dims(const std::string& name) const {
      return get(name).dims_.size();
    }
    var_origin variable_map::get_origin(const std::string& name) const {
      if (!exists(name))
        throw std::invalid_argument("variable does not exist");
      return map_.find(name)->second.second;
    }
    void variable_map::add(const std::string& name,
                           const base_var_decl& base_decl,
                           const var_origin& vo) {
      map_[name] = range_t(base_decl, vo);
    }
    void variable_map::remove(const std::string& name) {
      map_.erase(name);
    }

    int_var_decl::int_var_decl()
      : base_var_decl(INT_T)
    { }

    int_var_decl::int_var_decl(range const& range,
                               std::string const& name,
                               std::vector<expression> const& dims)
      : base_var_decl(name, dims, INT_T),
        range_(range)
    { }



    double_var_decl::double_var_decl()
      : base_var_decl(DOUBLE_T)
    { }

    double_var_decl::double_var_decl(range const& range,
                                     std::string const& name,
                                     std::vector<expression> const& dims)
      : base_var_decl(name, dims, DOUBLE_T),
        range_(range)
    { }

    unit_vector_var_decl::unit_vector_var_decl()
      : base_var_decl(VECTOR_T)
    { }

    unit_vector_var_decl::unit_vector_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, VECTOR_T),
        K_(K)
    { }

    simplex_var_decl::simplex_var_decl()
      : base_var_decl(VECTOR_T)
    { }

    simplex_var_decl::simplex_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, VECTOR_T),
        K_(K)
    { }

    ordered_var_decl::ordered_var_decl()
      : base_var_decl(VECTOR_T)
    { }

    ordered_var_decl::ordered_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims)
        : base_var_decl(name, dims, VECTOR_T),
          K_(K) {
      }

    positive_ordered_var_decl::positive_ordered_var_decl()
      : base_var_decl(VECTOR_T)
    { }

    positive_ordered_var_decl::positive_ordered_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims)
        : base_var_decl(name, dims, VECTOR_T),
          K_(K) {
      }

    vector_var_decl::vector_var_decl() : base_var_decl(VECTOR_T) { }

    vector_var_decl::vector_var_decl(range const& range,
                                     expression const& M,
                                     std::string const& name,
                                     std::vector<expression> const& dims)
        : base_var_decl(name, dims, VECTOR_T),
          range_(range),
          M_(M) {
    }

    row_vector_var_decl::row_vector_var_decl() : base_var_decl(ROW_VECTOR_T) { }
    row_vector_var_decl::row_vector_var_decl(range const& range,
                                        expression const& N,
                                        std::string const& name,
                                        std::vector<expression> const& dims)
        : base_var_decl(name, dims, ROW_VECTOR_T),
          range_(range),
          N_(N) {
    }

    matrix_var_decl::matrix_var_decl() : base_var_decl(MATRIX_T) { }
    matrix_var_decl::matrix_var_decl(range const& range,
                      expression const& M,
                      expression const& N,
                      std::string const& name,
                      std::vector<expression> const& dims)
        : base_var_decl(name, dims, MATRIX_T),
          range_(range),
          M_(M),
          N_(N) {
    }


    cholesky_factor_var_decl::cholesky_factor_var_decl()
      : base_var_decl(MATRIX_T) {
    }
    cholesky_factor_var_decl::cholesky_factor_var_decl(expression const& M,
                                       expression const& N,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, MATRIX_T),
        M_(M),
        N_(N) {
    }

    cholesky_corr_var_decl::cholesky_corr_var_decl()
      : base_var_decl(MATRIX_T) {
    }
    cholesky_corr_var_decl::cholesky_corr_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, MATRIX_T),
        K_(K) {
    }

    cov_matrix_var_decl::cov_matrix_var_decl() : base_var_decl(MATRIX_T) {
    }
    cov_matrix_var_decl::cov_matrix_var_decl(expression const& K,
                                         std::string const& name,
                                         std::vector<expression> const& dims)
      : base_var_decl(name, dims, MATRIX_T),
        K_(K) {
    }

    corr_matrix_var_decl::corr_matrix_var_decl() : base_var_decl(MATRIX_T) { }
    corr_matrix_var_decl::corr_matrix_var_decl(expression const& K,
                                   std::string const& name,
                                   std::vector<expression> const& dims)
        : base_var_decl(name, dims, MATRIX_T),
          K_(K) {
    }




    name_vis::name_vis() { }
    std::string name_vis::operator()(const nil& /* x */) const {
      return "";  // fail if arises
    }
    std::string name_vis::operator()(const int_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const double_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const vector_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const row_vector_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const matrix_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const unit_vector_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const simplex_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const ordered_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const positive_ordered_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const cholesky_factor_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const cholesky_corr_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const cov_matrix_var_decl& x) const {
      return x.name_;
    }
    std::string name_vis::operator()(const corr_matrix_var_decl& x) const {
      return x.name_;
    }

    var_decl_base_type_vis::var_decl_base_type_vis() { }
    base_var_decl var_decl_base_type_vis::operator()(const nil& /* x */)
      const {
      return base_var_decl();  // should not be called
    }
    base_var_decl var_decl_base_type_vis::operator()(const int_var_decl& x)
      const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(const double_var_decl& x)
      const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(const vector_var_decl& x)
      const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                    const row_vector_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(const matrix_var_decl& x)
      const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                    const unit_vector_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const simplex_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const ordered_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                             const positive_ordered_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const cholesky_factor_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const cholesky_corr_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const cov_matrix_var_decl& x) const {
      return x.base_type_;
    }
    base_var_decl var_decl_base_type_vis::operator()(
                                     const corr_matrix_var_decl& x) const {
      return x.base_type_;
    }



   // can't template out in .cpp file

    var_decl::var_decl(const var_decl_t& decl) : decl_(decl) { }
    var_decl::var_decl() : decl_(nil()) { }
    var_decl::var_decl(const nil& decl) : decl_(decl) { }
    var_decl::var_decl(const int_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const double_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const vector_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const row_vector_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const matrix_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const unit_vector_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const simplex_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const ordered_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const positive_ordered_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const cholesky_factor_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const cholesky_corr_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const cov_matrix_var_decl& decl) : decl_(decl) { }
    var_decl::var_decl(const corr_matrix_var_decl& decl) : decl_(decl) { }

    std::string var_decl::name() const {
      return boost::apply_visitor(name_vis(), decl_);
    }

    base_var_decl var_decl::base_decl() const {
      return boost::apply_visitor(var_decl_base_type_vis(), decl_);
    }

    statement::statement() : statement_(nil()) { }

    statement::statement(const statement_t& st) : statement_(st) { }
    statement::statement(const nil& st) : statement_(st) { }
    statement::statement(const assignment& st) : statement_(st) { }
    statement::statement(const assgn& st) : statement_(st) { }
    statement::statement(const sample& st) : statement_(st) { }
    statement::statement(const increment_log_prob_statement& st)
      : statement_(st) {
    }
    statement::statement(const statements& st) : statement_(st) { }
    statement::statement(const expression& st) : statement_(st) { }
    statement::statement(const for_statement& st) : statement_(st) { }
    statement::statement(const while_statement& st) : statement_(st) { }
    statement::statement(const break_continue_statement& st)
      : statement_(st) { }
    statement::statement(const conditional_statement& st) : statement_(st) { }
    statement::statement(const print_statement& st) : statement_(st) { }
    statement::statement(const reject_statement& st) : statement_(st) { }
    statement::statement(const return_statement& st) : statement_(st) { }
    statement::statement(const no_op_statement& st) : statement_(st) { }


    bool is_no_op_statement_vis::operator()(const nil& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const assignment& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const assgn& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const sample& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(
                            const increment_log_prob_statement& t) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const expression& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const statements& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const for_statement& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(
                                    const conditional_statement& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const while_statement& st) const {
      return false;
    }
    bool
    is_no_op_statement_vis::operator()(const break_continue_statement& st)
      const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const print_statement& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const reject_statement& st) const {
      return false;
    }
    bool is_no_op_statement_vis::operator()(const no_op_statement& st) const {
      return true;
    }
    bool is_no_op_statement_vis::operator()(const return_statement& st) const {
      return false;
    }

    bool statement::is_no_op_statement() const {
      is_no_op_statement_vis vis;
      return boost::apply_visitor(vis, statement_);
    }

    increment_log_prob_statement::increment_log_prob_statement() {
    }
    increment_log_prob_statement::increment_log_prob_statement(
                                               const expression& log_prob)
      : log_prob_(log_prob) {
    }

    for_statement::for_statement() {
    }
    for_statement::for_statement(std::string& variable,
                                 range& range,
                                 statement& stmt)
      : variable_(variable),
        range_(range),
        statement_(stmt) {
    }

    while_statement::while_statement() { }
    while_statement::while_statement(const expression& condition,
                                     const statement& body)
      : condition_(condition),
        body_(body) {
    }

    break_continue_statement::break_continue_statement() { }
    break_continue_statement::break_continue_statement(const std::string& s)
      : generate_(s) { }

    conditional_statement::conditional_statement() {
    }
    conditional_statement
    ::conditional_statement(const std::vector<expression>& conditions,
                            const std::vector<statement>& bodies)
      : conditions_(conditions),
        bodies_(bodies) {
    }

    return_statement::return_statement() { }
    return_statement::return_statement(const expression& expr)
      : return_value_(expr) {
    }

    print_statement::print_statement() { }

    print_statement::print_statement(const std::vector<printable>& printables)
      : printables_(printables) {
    }

    reject_statement::reject_statement() { }

    reject_statement::reject_statement(const std::vector<printable>& printables)
      : printables_(printables) {
    }

    program::program() { }
    program::program(const std::vector<function_decl_def>& function_decl_defs,
                     const std::vector<var_decl>& data_decl,
                     const std::pair<std::vector<var_decl>,
                     std::vector<statement> >& derived_data_decl,
                     const std::vector<var_decl>& parameter_decl,
                     const std::pair<std::vector<var_decl>,
                     std::vector<statement> >& derived_decl,
                     const statement& st,
                     const std::pair<std::vector<var_decl>,
                     std::vector<statement> >& generated_decl)
      : function_decl_defs_(function_decl_defs),
        data_decl_(data_decl),
        derived_data_decl_(derived_data_decl),
        parameter_decl_(parameter_decl),
        derived_decl_(derived_decl),
        statement_(st),
        generated_decl_(generated_decl) {
    }

    sample::sample() {
      }
    sample::sample(expression& e,
             distribution& dist)
        : expr_(e),
          dist_(dist) {
      }
    bool sample::is_ill_formed() const {
        return expr_.expression_type().is_ill_formed()
          || (truncation_.has_low()
              && expr_.expression_type() != truncation_.low_.expression_type())
          || (truncation_.has_high()
               && expr_.expression_type()
                  != truncation_.high_.expression_type());
    }

    assignment::assignment() {
    }
    assignment::assignment(variable_dims& var_dims,
                           expression& expr)
      : var_dims_(var_dims),
        expr_(expr) {
    }

    var_occurs_vis::var_occurs_vis(const variable& e)
      : var_name_(e.name_) {
    }

    bool var_occurs_vis::operator()(const nil& st) const {
      return false;
    }
    bool var_occurs_vis::operator()(const int_literal& e) const {
      return false;
    }
    bool var_occurs_vis::operator()(const double_literal& e) const {
      return false;
    }
    bool var_occurs_vis::operator()(const array_literal& e) const {
      return false;  // TODO(carpenter): update for array_literal
    }
    bool var_occurs_vis::operator()(const variable& e) const {
      return var_name_ == e.name_;
    }
    bool var_occurs_vis::operator()(const fun& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }
    bool var_occurs_vis::operator()(const integrate_ode& e) const {
      return false;  // no refs persist out of integrate_ode() call
    }
    bool var_occurs_vis::operator()(const integrate_ode_control& e) const {
      return false;  // no refs persist out of integrate_ode_control() call
    }
    bool var_occurs_vis::operator()(const index_op& e) const {
      // refs only persist out of expression, not indexes
      return boost::apply_visitor(*this, e.expr_.expr_);
    }
    bool var_occurs_vis::operator()(const index_op_sliced& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }
    bool var_occurs_vis::operator()(const conditional_op& e) const {
      return boost::apply_visitor(*this, e.cond_.expr_)
        || boost::apply_visitor(*this, e.true_val_.expr_)
        || boost::apply_visitor(*this, e.false_val_.expr_);
    }
    bool var_occurs_vis::operator()(const binary_op& e) const {
      return boost::apply_visitor(*this, e.left.expr_)
        || boost::apply_visitor(*this, e.right.expr_);
    }
    bool var_occurs_vis::operator()(const unary_op& e) const {
      return boost::apply_visitor(*this, e.subject.expr_);
    }

    assgn::assgn() { }
    assgn::assgn(const variable& lhs_var, const std::vector<idx>& idxs,
                 const expression& rhs)
      : lhs_var_(lhs_var), idxs_(idxs), rhs_(rhs) { }

    bool assgn::lhs_var_occurs_on_rhs() const {
      var_occurs_vis vis(lhs_var_);
      return boost::apply_visitor(vis, rhs_.expr_);
    }

    /**
     * Return the expression type for the result of applying the
     * specified indexes to the specified expression.  If the reuslt
     * is ill typed, the output type is <code>ILL_FORMED_T</code> and
     * dimensions are <code>OU</code>.
     *
     * @param[in] e Expression to index.
     * @param[in] idxs Vector of indexes.
     * @return Type of indexed expression.
     */
    expr_type indexed_type(const expression& e,
                           const std::vector<idx>& idxs) {
      expr_type e_type = e.expression_type();

      base_expr_type base_type = e_type.base_type_;
      size_t base_dims = e_type.num_dims_;
      size_t unindexed_dims = base_dims;
      size_t out_dims = 0U;
      size_t i = 0;
      for ( ; unindexed_dims > 0 && i < idxs.size(); ++i, --unindexed_dims)
        if (is_multi_index(idxs[i]))
          ++out_dims;
      if (idxs.size() - i == 0) {
        return expr_type(base_type, out_dims + unindexed_dims);
      } else if (idxs.size() - i == 1) {
        if (base_type == MATRIX_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(MATRIX_T, out_dims);
          else
            return expr_type(ROW_VECTOR_T, out_dims);
        } else if (base_type == VECTOR_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else if (base_type == ROW_VECTOR_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(ROW_VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else {
          return expr_type(ILL_FORMED_T, 0U);
        }
      } else if (idxs.size() - i == 2) {
        if (base_type == MATRIX_T) {
          if (is_multi_index(idxs[i]) && is_multi_index(idxs[i + 1]))
            return expr_type(MATRIX_T, out_dims);
          else if (is_multi_index(idxs[i]))
            return expr_type(VECTOR_T, out_dims);
          else if (is_multi_index(idxs[i + 1]))
            return expr_type(ROW_VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else {
          return expr_type(ILL_FORMED_T, 0U);
        }
      } else {
        return expr_type(ILL_FORMED_T, 0U);
      }
    }


    expression& expression::operator+=(const expression& rhs) {
      expr_ = binary_op(expr_, "+", rhs);
      return *this;
    }

    expression& expression::operator-=(const expression& rhs) {
      expr_ = binary_op(expr_, "-", rhs);
      return *this;
    }

    expression& expression::operator*=(expression const& rhs) {
      expr_ = binary_op(expr_, "*", rhs);
      return *this;
    }

    expression& expression::operator/=(expression const& rhs) {
      expr_ = binary_op(expr_, "/", rhs);
      return *this;
    }

    bool has_rng_suffix(const std::string& s) {
      int n = s.size();
      return n > 4
        && s[n-1] == 'g'
        && s[n-2] == 'n'
        && s[n-3] == 'r'
        && s[n-4] == '_';
    }

    bool has_lp_suffix(const std::string& s) {
      int n = s.size();
      return n > 3
        && s[n-1] == 'p'
        && s[n-2] == 'l'
        && s[n-3] == '_';
    }

    bool is_user_defined(const std::string& name,
                         const std::vector<expression>& args) {
      std::vector<expr_type> arg_types;
      for (size_t i = 0; i <  args.size(); ++i)
        arg_types.push_back(args[i].expression_type());
      function_signature_t sig;
      int matches
        = function_signatures::instance()
        .get_signature_matches(name, arg_types, sig);
      if (matches != 1)
        return false;  // reall shouldn't come up;  throw instead?
      std::pair<std::string, function_signature_t>
        name_sig(name, sig);
      return function_signatures::instance().is_user_defined(name_sig);
    }

    bool is_user_defined_prob_function(const std::string& name,
                                       const expression& variate,
                                       const std::vector<expression>& params) {
      std::vector<expression> variate_params;
      variate_params.push_back(variate);
      for (size_t i = 0; i < params.size(); ++i)
        variate_params.push_back(params[i]);
      return is_user_defined(name, variate_params);
    }

    bool is_user_defined(const fun& fx) {
      return is_user_defined(fx.name_, fx.args_);
    }

    bool is_assignable(const expr_type& l_type,
                       const expr_type& r_type,
                       const std::string& failure_message,
                       std::ostream& error_msgs) {
      bool assignable = true;
      if (l_type.num_dims_ != r_type.num_dims_) {
        assignable = false;
        error_msgs << "Mismatched array dimensions.";
      }
      if (l_type.base_type_ != r_type.base_type_
          && (!(l_type.base_type_ == DOUBLE_T
                && r_type.base_type_ == INT_T))) {
        assignable = false;
        error_msgs << "Base type mismatch. ";
      }
      if (!assignable)
        error_msgs << failure_message
                   << std::endl
                   << "    LHS type = " << l_type
                   << "; RHS type = " << r_type
                   << std::endl;
      return assignable;
    }

    bool ends_with(const std::string& suffix,
                   const std::string& s) {
      size_t idx = s.rfind(suffix);
      return idx != std::string::npos
        && idx == (s.size() - suffix.size());
    }

   std::string get_cdf(const std::string& dist_name) {
      if (function_signatures::instance().has_key(dist_name + "_cdf_log"))
        return dist_name + "_cdf_log";
      else if (function_signatures::instance().has_key(dist_name + "_lcdf"))
        return dist_name + "_lcdf";
      else
        return dist_name;
    }

    std::string get_ccdf(const std::string& dist_name) {
      if (function_signatures::instance().has_key(dist_name + "_ccdf_log"))
        return dist_name + "_ccdf_log";
      else if (function_signatures::instance().has_key(dist_name + "_lccdf"))
        return dist_name + "_lccdf";
      else
        return dist_name;
    }

    std::string get_prob_fun(const std::string& dist_name) {
      if (function_signatures::instance().has_key(dist_name + "_log"))
        return dist_name + "_log";
      else if (function_signatures::instance().has_key(dist_name + "_lpdf"))
        return dist_name + "_lpdf";
      else if (function_signatures::instance().has_key(dist_name + "_lpmf"))
        return dist_name + "_lpmf";
      else
        return dist_name;
    }

    bool has_prob_fun_suffix(const std::string& fname) {
      return ends_with("_lpdf", fname) || ends_with("_lpmf", fname)
        || ends_with("_log", fname);
    }

    std::string strip_prob_fun_suffix(const std::string& fname) {
      if (ends_with("_lpdf", fname))
        return fname.substr(0, fname.size() - 5);
      else if (ends_with("_lpmf", fname))
        return fname.substr(0, fname.size() - 5);
      else if (ends_with("_log", fname))
        return fname.substr(0, fname.size() - 4);
      else
        return fname;
    }

    bool has_cdf_suffix(const std::string& fname) {
      return ends_with("_lcdf", fname) || ends_with("_cdf_log", fname);
    }

    std::string strip_cdf_suffix(const std::string& fname) {
      if (ends_with("_lcdf", fname))
        return fname.substr(0, fname.size() - 5);
      else if (ends_with("_cdf_log", fname))
        return fname.substr(0, fname.size() - 8);
      else
        return fname;
    }

    bool has_ccdf_suffix(const std::string& fname) {
      return ends_with("_lccdf", fname) || ends_with("_ccdf_log", fname);
    }

    std::string strip_ccdf_suffix(const std::string& fname) {
      if (ends_with("_lccdf", fname))
        return fname.substr(0, fname.size() - 6);
      else if (ends_with("_ccdf_log", fname))
        return fname.substr(0, fname.size() - 9);
      else
        return fname;
    }

    bool fun_name_exists(const std::string& name) {
      return function_signatures::instance().has_key(name);
    }



  }
}
#endif
