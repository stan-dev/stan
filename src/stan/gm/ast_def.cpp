#ifndef __STAN__GM__AST_DEF_HPP__
#define __STAN__GM__AST_DEF_HPP__

#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <cstddef>
#include <limits>
#include <climits>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stan/gm/ast.hpp>

namespace stan {

  namespace gm {

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
        || ( base_type_ == et.base_type_
             && num_dims_ < et.num_dims_ );
    }
    bool expr_type::operator<=(const expr_type& et) const {
      return (base_type_ < et.base_type_)
        || ( base_type_ == et.base_type_
             && num_dims_ <= et.num_dims_ );
    }
    bool expr_type::operator>(const expr_type& et) const {
      return (base_type_ > et.base_type_)
        || ( base_type_ == et.base_type_
             && num_dims_ > et.num_dims_ );
    }
    bool expr_type::operator>=(const expr_type& et) const {
      return (base_type_ > et.base_type_)
        || ( base_type_ == et.base_type_
             && num_dims_ >= et.num_dims_ );
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

    std::ostream& operator<<(std::ostream& o, const expr_type& et) {
      write_base_expr_type(o,et.type());
      if (et.num_dims() > 0) 
        o << '[' << et.num_dims() << ']';
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
    bool
    function_signatures::is_defined(const std::string& name, 
                                    const function_signature_t& sig) {
      const std::vector<function_signature_t> sigs = sigs_map_[name];
      for (size_t i = 0; i < sigs.size(); ++i)
        if (sig.second  == sigs[i].second)
          return true;
      return false;
    }
    void function_signatures::add(const std::string& name,
                                   const expr_type& result_type,
                                   const std::vector<expr_type>& arg_types) {
      sigs_map_[name].push_back(function_signature_t(result_type,arg_types));

    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type) {
      std::vector<expr_type> arg_types;
      add(name,result_type,arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type);
      add(name,result_type,arg_types);
    }
    void function_signatures::add(const std::string& name,
                                  const expr_type& result_type,
                                  const expr_type& arg_type1,
                                  const expr_type& arg_type2) {
      std::vector<expr_type> arg_types;
      arg_types.push_back(arg_type1);
      arg_types.push_back(arg_type2);
      add(name,result_type,arg_types);
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
      add(name,result_type,arg_types);
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
      add(name,result_type,arg_types);
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
      add(name,result_type,arg_types);
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
      add(name,result_type,arg_types);
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
      add(name,result_type,arg_types);
    }
    void function_signatures::add_nullary(const::std::string& name) {
      add(name,DOUBLE_T);
    }
    void function_signatures::add_unary(const::std::string& name) {
      add(name,DOUBLE_T,DOUBLE_T);
    }
    void function_signatures::add_binary(const::std::string& name) {
      add(name,DOUBLE_T,DOUBLE_T,DOUBLE_T);
    }
    void function_signatures::add_ternary(const::std::string& name) {
      add(name,DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T);
    }
    void function_signatures::add_quaternary(const::std::string& name) {
      add(name,DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T);
    }
    int function_signatures::num_promotions(
                            const std::vector<expr_type>& call_args,
                            const std::vector<expr_type>& sig_args) {
      if (call_args.size() != sig_args.size()) {
        return -1; // failure
      }
      int num_promotions = 0;
      for (size_t i = 0; i < call_args.size(); ++i) {
        if (call_args[i] == sig_args[i]) {
          continue;
        } else if (call_args[i].is_primitive_int()
                   && sig_args[i].is_primitive_double()) {
          ++num_promotions;
        } else {
          return -1; // failed match
        } 
      }
      return num_promotions;
    }
    int function_signatures::get_signature_matches(const std::string& name,
                              const std::vector<expr_type>& args,
                              function_signature_t& signature) {

      std::vector<function_signature_t> signatures = sigs_map_[name];
      size_t min_promotions = std::numeric_limits<size_t>::max(); 
      size_t num_matches = 0;
      for (size_t i = 0; i < signatures.size(); ++i) {
        signature = signatures[i];
        int promotions = num_promotions(args,signature.second);
        if (promotions < 0) continue; // no match
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
        
    expr_type function_signatures::get_result_type(
                                         const std::string& name,
                                         const std::vector<expr_type>& args,
                                         std::ostream& error_msgs) {
      std::vector<function_signature_t> signatures = sigs_map_[name];
      size_t match_index = 0; 
      size_t min_promotions = std::numeric_limits<size_t>::max(); 
      size_t num_matches = 0;

      for (size_t i = 0; i < signatures.size(); ++i) {
        int promotions = num_promotions(args,signatures[i].second);
        if (promotions < 0) continue; // no match
        size_t promotions_ui = static_cast<size_t>(promotions);
        if (promotions_ui < min_promotions) {
          min_promotions = promotions_ui;
          match_index = i;
          num_matches = 1;
        } else if (promotions_ui == min_promotions) {
          ++num_matches;
        }
      }

      if (num_matches == 1) {
        return signatures[match_index].first;
      } else if (num_matches == 0) {
        error_msgs << "no matches for function name=\"" << name << "\"" 
                   << std::endl;
      } else {
        error_msgs << num_matches << " matches with " 
                   << min_promotions << " integer promotions "
                   << "for function name=\"" << name << "\"" << std::endl;
      }
      for (size_t i = 0; i < args.size(); ++i)
        error_msgs << "    arg " << i << " type=" << args[i] << std::endl;

      error_msgs << "available function signatures for "
                 << name << ":" << std::endl;
      for (size_t i = 0; i < signatures.size(); ++i) {
        error_msgs << i << ".  " << name << "(";
        for (size_t j = 0; j < signatures[i].second.size(); ++j) {
          if (j > 0) error_msgs << ", ";
          error_msgs << signatures[i].second[j];
        }
        error_msgs << ") : " << signatures[i].first << std::endl;
      }
      return expr_type(); // ill-formed dummy
    }
    function_signatures::function_signatures() { 
#include <stan/gm/function_signatures.h>
    }
    std::set<std::string>
    function_signatures::key_set() const {
      using std::map;
      using std::set;
      using std::string;
      using std::vector;
      set<string> result;
      for (map<string,vector<function_signature_t> >::const_iterator 
             it = sigs_map_.begin();
           it != sigs_map_.end();
           ++it)
        result.insert(it->first);
      return result;
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
        dims.push_back(expression(int_literal(0))); // dummy value 0
      return base_var_decl(name_,dims,arg_type_.base_type_);
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
        error_msgs_ << ( "Expecting return, found"
                         " statement sequence with empty body." )
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
    bool returns_type_vis::operator()(const 
                                      conditional_statement& st) const  {
      // all condition bodies must end in appropriate return
      if (st.bodies_.size() != (st.conditions_.size() + 1)) {
        error_msgs_ << ( "Expecting return, found conditional"
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
      returns_type_vis vis(return_type,error_msgs);
      return boost::apply_visitor(vis,statement.statement_);
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
      return expr_type(DOUBLE_T,2);
    }
    expr_type expression_type_vis::operator()(const fun& e) const {
      return e.type_;
    }
    expr_type expression_type_vis::operator()(const index_op& e) const {
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
      return boost::apply_visitor(vis,expr_);
    }
    // template <typename Expr>
    // expression::expression(const Expr& expr) : expr_(expr) {  }

    expression::expression(const expression_t& expr) : expr_(expr) { }
    expression::expression(const nil& expr) : expr_(expr) { }
    expression::expression(const int_literal& expr) : expr_(expr) { }
    expression::expression(const double_literal& expr) : expr_(expr) { }
    expression::expression(const array_literal& expr) : expr_(expr) { }
    expression::expression(const variable& expr) : expr_(expr) { }
    expression::expression(const integrate_ode& expr) : expr_(expr) { }
    expression::expression(const fun& expr) : expr_(expr) { }
    expression::expression(const index_op& expr) : expr_(expr) { }
    expression::expression(const binary_op& expr) : expr_(expr) { }
    expression::expression(const unary_op& expr) : expr_(expr) { }

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
        if (boost::apply_visitor(*this,e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_var::operator()(const variable& e) const {
      var_origin vo = var_map_.get_origin(e.name_);
      return ( vo == parameter_origin
               || vo == transformed_parameter_origin
               || vo == local_origin );
    }
    bool contains_var::operator()(const fun& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this,e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_var::operator()(const integrate_ode& e) const {
      // only init state and params may contain vars
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_)
        ;
    }
    bool contains_var::operator()(const index_op& e) const {
      return boost::apply_visitor(*this,e.expr_.expr_);
    }
    bool contains_var::operator()(const binary_op& e) const {
      return boost::apply_visitor(*this,e.left.expr_)
        || boost::apply_visitor(*this,e.right.expr_);
    }
    bool contains_var::operator()(const unary_op& e) const {
        return boost::apply_visitor(*this,e.subject.expr_);
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
        || name == "transpose"
        ;
    }

    bool has_var(const expression& e,
                 const variable_map& var_map) {
      contains_var vis(var_map);
      return boost::apply_visitor(vis,e.expr_);
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
        if (boost::apply_visitor(*this,e.args_[i].expr_))
          return true;
      return false;
    }
    bool contains_nonparam_var::operator()(const variable& e) const {
      var_origin vo = var_map_.get_origin(e.name_);
      return ( vo == transformed_parameter_origin
               || vo == local_origin );
    }
    bool contains_nonparam_var::operator()(const integrate_ode& e) const {
      // if any vars, return true because integration will be nonlinear
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_)
        ;
    }
    bool contains_nonparam_var::operator()(const fun& e) const {
      // any function applied to non-linearly transformed var
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this,e.args_[i].expr_))
          return true;
      // non-linear function applied to var
      if (!is_linear_function(e.name_)) {
        for (size_t i = 0; i < e.args_.size(); ++i)
          if (has_var(e.args_[i],var_map_))
            return true;
      }
      return false;
    }
    bool contains_nonparam_var::operator()(const index_op& e) const {
      return boost::apply_visitor(*this,e.expr_.expr_);
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
      if (has_non_param_var(e.left,var_map_)
          || has_non_param_var(e.right,var_map_))
        return true;
      if (e.op == "*" || e.op == "/")
        return has_var(e.left,var_map_) && has_var(e.right,var_map_);
      return false;
    }
    bool contains_nonparam_var::operator()(const unary_op& e) const {
      // only negation, which is linear, so recurse
      return has_non_param_var(e.subject,var_map_);
    }

    bool has_non_param_var(const expression& e,
                           const variable_map& var_map) {
      contains_nonparam_var vis(var_map);
      return boost::apply_visitor(vis,e.expr_);
    }

    bool is_nil_op::operator()(const nil& /*x*/) const { return true; }
    bool is_nil_op::operator()(const int_literal& /*x*/) const { return false; }
    bool is_nil_op::operator()(const double_literal& /* x */) const { return false; }
    bool is_nil_op::operator()(const array_literal& /* x */) const { return false; }
    bool is_nil_op::operator()(const variable& /* x */) const { return false; }
    bool is_nil_op::operator()(const integrate_ode& /* x */) const { return false; }
    bool is_nil_op::operator()(const fun& /* x */) const { return false; }
    bool is_nil_op::operator()(const index_op& /* x */) const { return false; }
    bool is_nil_op::operator()(const binary_op& /* x */) const { return false; }
    bool is_nil_op::operator()(const unary_op& /* x */) const { return false; }
      
    // template <typename T>
    // bool is_nil_op::operator()(const T& /* x */) const { return false; }

    bool is_nil(const expression& e) {
      is_nil_op ino;
      return boost::apply_visitor(ino,e.expr_);
    }

    variable_dims::variable_dims() { } // req for FUSION_ADAPT
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
      : type_(DOUBLE_T,0U) { 
    }
    double_literal::double_literal(double val)
      : val_(val),
        type_(DOUBLE_T,0U) {
    }
    double_literal& double_literal::operator=(const double_literal& dl) {
      val_ = dl.val_;
      type_ = dl.type_;
      return *this;
    }


    array_literal::array_literal() 
      : args_(),
        type_(DOUBLE_T,1U) {
    }
    array_literal::array_literal(const std::vector<expression>& args) 
      : args_(args),
        type_() { // ill-formed w/o help
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
    integrate_ode::integrate_ode(const std::string& system_function_name,
                         const expression& y0,
                         const expression& t0,
                         const expression& ts,
                         const expression& theta,
                         const expression& x,
                         const expression& x_int) 
      : system_function_name_(system_function_name),
        y0_(y0),
        t0_(t0),
        ts_(ts),
        theta_(theta),
        x_(x),
        x_int_(x_int) {
    }


    fun::fun() { }
    fun::fun(std::string const& name,
             std::vector<expression> const& args) 
      : name_(name),
        args_(args) {
      infer_type();
    }
    void fun::infer_type() {
      // FIXME: remove this useless function and any calls to it
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
        return expr_type(expr_base_type,num_expr_dims - num_index_dims);
      if (num_index_dims == (num_expr_dims + 1)) {
        if (expr_base_type == VECTOR_T || expr_base_type == ROW_VECTOR_T)
          return expr_type(DOUBLE_T,0U);
        if (expr_base_type == MATRIX_T)
          return expr_type(ROW_VECTOR_T,0U);
      }
      if (num_index_dims == (num_expr_dims + 2))
        if (expr_base_type == MATRIX_T)
          return expr_type(DOUBLE_T,0U);
      
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
      type_ = infer_type_indexing(expr_,total_dims(dimss_));
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
      : name_(name),
          dims_(dims),
        base_type_(base_type) {
    }

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
      map_[name] = range_t(base_decl,vo);
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
      : base_var_decl(name,dims,INT_T),
        range_(range)
    { }
    


    double_var_decl::double_var_decl() 
      : base_var_decl(DOUBLE_T) 
    { }

    double_var_decl::double_var_decl(range const& range,
                                     std::string const& name,
                                     std::vector<expression> const& dims)
      : base_var_decl(name,dims,DOUBLE_T),
        range_(range) 
    { }

    unit_vector_var_decl::unit_vector_var_decl() 
      : base_var_decl(VECTOR_T) 
    { }

    unit_vector_var_decl::unit_vector_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name,dims,VECTOR_T),
        K_(K) 
    { }

    simplex_var_decl::simplex_var_decl() 
      : base_var_decl(VECTOR_T) 
    { }

    simplex_var_decl::simplex_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name,dims,VECTOR_T),
        K_(K) 
    { }

    ordered_var_decl::ordered_var_decl() 
      : base_var_decl(VECTOR_T) 
    { }

    ordered_var_decl::ordered_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims)
        : base_var_decl(name,dims,VECTOR_T),
          K_(K) {
      }
    
    positive_ordered_var_decl::positive_ordered_var_decl() 
      : base_var_decl(VECTOR_T) 
    { }

    positive_ordered_var_decl::positive_ordered_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims)
        : base_var_decl(name,dims,VECTOR_T),
          K_(K) {
      }
    
    vector_var_decl::vector_var_decl() : base_var_decl(VECTOR_T) { }

    vector_var_decl::vector_var_decl(range const& range,
                                     expression const& M,
                                     std::string const& name,
                                     std::vector<expression> const& dims)
        : base_var_decl(name,dims,VECTOR_T),
          range_(range),
          M_(M) {
    }
    
    row_vector_var_decl::row_vector_var_decl() : base_var_decl(ROW_VECTOR_T) { }
    row_vector_var_decl::row_vector_var_decl(range const& range,
                                        expression const& N,
                                        std::string const& name,
                                        std::vector<expression> const& dims)
        : base_var_decl(name,dims,ROW_VECTOR_T),
          range_(range),
          N_(N) {
    }

    matrix_var_decl::matrix_var_decl() : base_var_decl(MATRIX_T) { }
    matrix_var_decl::matrix_var_decl(range const& range,
                      expression const& M,
                      expression const& N,
                      std::string const& name,
                      std::vector<expression> const& dims)
        : base_var_decl(name,dims,MATRIX_T),
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
      : base_var_decl(name,dims,MATRIX_T),
        M_(M),
        N_(N) {
    }

    cholesky_corr_var_decl::cholesky_corr_var_decl() 
      : base_var_decl(MATRIX_T) { 
    }
    cholesky_corr_var_decl::cholesky_corr_var_decl(expression const& K,
                                                   std::string const& name,
                                                   std::vector<expression> const& dims)
      : base_var_decl(name,dims,MATRIX_T),
        K_(K) {
    }

    cov_matrix_var_decl::cov_matrix_var_decl() : base_var_decl(MATRIX_T) { 
    }
    cov_matrix_var_decl::cov_matrix_var_decl(expression const& K,
                                             std::string const& name,
                                             std::vector<expression> const& dims)
      : base_var_decl(name,dims,MATRIX_T),
        K_(K) {
    }

    corr_matrix_var_decl::corr_matrix_var_decl() : base_var_decl(MATRIX_T) { }
    corr_matrix_var_decl::corr_matrix_var_decl(expression const& K,
                                   std::string const& name,
                                   std::vector<expression> const& dims)
        : base_var_decl(name,dims,MATRIX_T),
          K_(K) {
    }




    name_vis::name_vis() { }
    std::string name_vis::operator()(const nil& /* x */) const { 
      return ""; // fail if arises
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
      return boost::apply_visitor(name_vis(),decl_);
    }

    statement::statement() : statement_(nil()) { }

    statement::statement(const statement_t& st) : statement_(st) { }
    statement::statement(const nil& st) : statement_(st) { }
    statement::statement(const assignment& st) : statement_(st) { }
    statement::statement(const sample& st) : statement_(st) { }
    statement::statement(const increment_log_prob_statement& st) : statement_(st) { }
    statement::statement(const statements& st) : statement_(st) { }
    statement::statement(const expression& st) : statement_(st) { }
    statement::statement(const for_statement& st) : statement_(st) { }
    statement::statement(const while_statement& st) : statement_(st) { }
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
    bool is_no_op_statement_vis::operator()(const sample& st) const { 
      return false; 
    }
    bool is_no_op_statement_vis::operator()(const increment_log_prob_statement& t) const {
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
    bool is_no_op_statement_vis::operator()(const conditional_statement& st) const { 
      return false; 
    }
    bool is_no_op_statement_vis::operator()(const while_statement& st) const { 
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
      return boost::apply_visitor(vis,statement_);
    }

    increment_log_prob_statement::increment_log_prob_statement() {
    }
    increment_log_prob_statement::increment_log_prob_statement(const expression& log_prob)
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

    while_statement::while_statement() { 
    }
    while_statement::while_statement(const expression& condition,
                                     const statement& body) 
      : condition_(condition),
        body_(body) {
    }

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
          || ( truncation_.has_low()
               && expr_.expression_type() 
                  != truncation_.low_.expression_type() )
          || ( truncation_.has_high()
               && expr_.expression_type() 
                  != truncation_.high_.expression_type() );
      }

    assignment::assignment() {
    }
    assignment::assignment(variable_dims& var_dims,
                           expression& expr)
      : var_dims_(var_dims),
        expr_(expr) {
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
        .get_signature_matches(name,arg_types,sig);
      if (matches != 1)
        return false; // reall shouldn't come up;  throw instead?
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
      return is_user_defined(name,variate_params);
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
          && (! (l_type.base_type_ == DOUBLE_T && r_type.base_type_ == INT_T))) {
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

  }
}


#endif
