#ifndef STAN__GM__PARSER__STATEMENT_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__STATEMENT_GRAMMAR_DEF__HPP

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>


#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/common_adaptors_def.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::gm::assignment,
                          (stan::gm::variable_dims, var_dims_)
                          (stan::gm::expression, expr_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::variable_dims,
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::distribution,
                          (std::string, family_)
                          (std::vector<stan::gm::expression>, args_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::for_statement,
                          (std::string, variable_)
                          (stan::gm::range, range_)
                          (stan::gm::statement, statement_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::return_statement,
                          (stan::gm::expression, return_value_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::print_statement,
                          (std::vector<stan::gm::printable>, printables_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::reject_statement,
                          (std::vector<stan::gm::printable>, printables_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::increment_log_prob_statement,
                          (stan::gm::expression, log_prob_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::sample,
                          (stan::gm::expression, expr_)
                          (stan::gm::distribution, dist_) 
                          (stan::gm::range, truncation_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::statements,
                          (std::vector<stan::gm::var_decl>, local_decl_)
                          (std::vector<stan::gm::statement>, statements_) );

namespace stan {

  namespace gm {

    struct validate_return_allowed {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(var_origin origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (origin != function_argument_origin
            && origin != function_argument_origin_lp
            && origin != function_argument_origin_rng) {
          error_msgs << "Returns only allowed from function bodies." << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_return_allowed> validate_return_allowed_f;

    struct validate_void_return_allowed {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(var_origin origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (origin != void_function_argument_origin
            && origin != void_function_argument_origin_lp
            && origin != void_function_argument_origin_rng) {
          error_msgs << "Void returns only allowed from function bodies of void return type." 
                     << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_void_return_allowed> validate_void_return_allowed_f;


    struct validate_assignment {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };

      bool operator()(assignment& a,
                      const var_origin& origin_allowed,
                      variable_map& vm,
                      std::ostream& error_msgs) const {

        // validate existence
        std::string name = a.var_dims_.name_;
        if (!vm.exists(name)) {
          error_msgs << "unknown variable in assignment"
                     << "; lhs variable=" << a.var_dims_.name_ 
                     << std::endl;
          return false;
        }

        // validate origin
        var_origin lhs_origin = vm.get_origin(name);
        if (lhs_origin != local_origin
            && lhs_origin != origin_allowed) {
          error_msgs << "attempt to assign variable in wrong block."
                     << " left-hand-side variable origin=";
          print_var_origin(error_msgs,lhs_origin);
          error_msgs << std::endl;
          return false;
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
          return false;
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
          return false;
        }

        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
        // allow int -> double promotion
        bool types_compatible 
          = lhs_base_type == rhs_base_type
          || ( lhs_base_type == DOUBLE_T && rhs_base_type == INT_T );
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; variable name = "
                     << a.var_dims_.name_
                     << ", type = ";
          write_base_expr_type(error_msgs,lhs_base_type);
          error_msgs << "; right-hand side type=";
          write_base_expr_type(error_msgs,rhs_base_type);
          error_msgs << std::endl;
          return false;
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
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_assignment> validate_assignment_f;

    struct validate_sample {
      template <typename T1, typename T2, typename T3>
      struct result { typedef bool type; };

      bool is_double_return(const std::string& function_name,
                            const std::vector<expr_type>& arg_types,
                            std::ostream& error_msgs) const {
        return function_signatures::instance()
          .get_result_type(function_name,arg_types,error_msgs)
          .is_primitive_double();
      }
      static bool is_univariate(const expr_type& et) {
        return et.num_dims_ == 0
          && ( et.base_type_ == INT_T
               || et.base_type_ == DOUBLE_T );
      }
      bool operator()(const sample& s,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const {
        std::vector<expr_type> arg_types;
        arg_types.push_back(s.expr_.expression_type());
        for (size_t i = 0; i < s.dist_.args_.size(); ++i)
          arg_types.push_back(s.dist_.args_[i].expression_type());
        std::string function_name(s.dist_.family_);
        function_name += "_log";
        if (!is_double_return(function_name,arg_types,error_msgs)) {
          error_msgs << "unknown distribution=" << s.dist_.family_ << std::endl;
          return false;
        }

        if (function_name == "lkj_cov_log") {
          error_msgs << "Warning: the lkj_cov_log() sampling distribution"
                     << " is deprecated.  It will be removed in Stan 3."
                     << std::endl
                     << "Code LKJ covariance in terms of an lkj_corr()"
                     << " distribution on a correlation matrix"
                     << " and independent lognormals on the scales."
                     << std::endl << std::endl;

        }

        // test for LHS not being purely a variable
        if (has_non_param_var(s.expr_,var_map)) {
          // FIXME:  really want to get line numbers in here too
          error_msgs << "Warning (non-fatal):"
             << " Left-hand side of sampling statement (~) contains a non-linear"
             << " transform of a parameter or local variable."
             << std::endl
             << " You must call increment_log_prob() with the log absolute determinant"
             << " of the Jacobian of the transform."
             << std::endl
             << "  Sampling Statement left-hand-side expression:"
             << std::endl
             << "    ";
          generate_expression(s.expr_,error_msgs);
          error_msgs << " ~ ";
          error_msgs << function_name << "(...)";
          error_msgs << std::endl;
        }
        // validate that variable and params are univariate if truncated
        if (s.truncation_.has_low() || s.truncation_.has_high()) {
          if (!is_univariate(s.expr_.expression_type())) { // .num_dims_ > 0) {
            error_msgs << "Outcomes in truncated distributions must be univariate."
                       << std::endl
                       << "  Found outcome expression: ";
            generate_expression(s.expr_,error_msgs);
            error_msgs << std::endl
                       << "  with non-univariate type: "
                       << s.expr_.expression_type()
                       << std::endl;
            return false;
          }
          for (size_t i = 0; i < s.dist_.args_.size(); ++i)
            if (!is_univariate(s.dist_.args_[i].expression_type())) { // .num_dims_ > 0) {
              error_msgs << "Parameters in truncated distributions must be univariate."
                         << std::endl
                         << "  Found parameter expression: ";
              generate_expression(s.dist_.args_[i],error_msgs);
              error_msgs << std::endl
                         << "  with non-univariate type: "
                         << s.dist_.args_[i].expression_type()
                         << std::endl;
              return false;
            }
        }
        if (s.truncation_.has_low()
            && !is_univariate(s.truncation_.low_.expression_type())) {
          error_msgs << "Lower boundsin truncated distributions must be univariate."
                     << std::endl
                     << "  Found lower bound expression: ";
          generate_expression(s.truncation_.low_,error_msgs);
          error_msgs << std::endl
                     << "  with non-univariate type: "
                     << s.truncation_.low_.expression_type()
                     << std::endl;
          return false;
        }
        if (s.truncation_.has_high() 
            && !is_univariate(s.truncation_.high_.expression_type())) {
          error_msgs << "Upper bounds in truncated distributions must be univariate."
                     << std::endl
                     << "  Found upper bound expression: ";
          generate_expression(s.truncation_.high_,error_msgs);
          error_msgs << std::endl
                     << "  with non-univariate type: "
                     << s.truncation_.high_.expression_type()
                     << std::endl;
          return false;
        }

        if (s.truncation_.has_low()) {
          std::vector<expr_type> arg_types_trunc(arg_types);
          arg_types_trunc[0] = s.truncation_.low_.expression_type(); 
          std::string function_name_cdf(s.dist_.family_);
          function_name_cdf += "_cdf_log";
          if (!is_double_return(function_name_cdf,arg_types_trunc,error_msgs)) {
            error_msgs << "lower truncation not defined for specified arguments to "
                       << s.dist_.family_ << std::endl;
            return false;
          }
          if (!is_double_return(function_name_cdf,arg_types,error_msgs)) {
            error_msgs << "lower bound in truncation type does not match"
                       << " sampled variate in distribution's type"
                       << std::endl;
            return false;
          }
        }
        if (s.truncation_.has_high()) {
          std::vector<expr_type> arg_types_trunc(arg_types);
          arg_types_trunc[0] = s.truncation_.high_.expression_type();
          std::string function_name_cdf(s.dist_.family_);
          function_name_cdf += "_cdf_log";
          if (!is_double_return(function_name_cdf,arg_types_trunc,error_msgs)) {
            error_msgs << "upper truncation not defined for specified arguments to "
                       << s.dist_.family_ << std::endl;
            return false;
          }
          if (!is_double_return(function_name_cdf,arg_types,error_msgs)) {
            error_msgs << "upper bound in truncation type does not match"
                       << " sampled variate in distribution's type" 
                       << std::endl;
            return false;
          }
        }
        return true;

      }
    };
    boost::phoenix::function<validate_sample> validate_sample_f;

    struct expression_as_statement {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(bool& pass,
                      const stan::gm::expression& expr,
                      std::stringstream& error_msgs) const {
        if (expr.expression_type() != VOID_T) {
          error_msgs << "Illegal statement beginning with non-void expression parsed as"
                     << std::endl << "  ";
          generate_expression(expr.expr_,error_msgs);
          error_msgs << std::endl
                     << "Not a legal assignment, sampling, or function statement.  Note that"
                     << std::endl
                     << "  * Assignment statements only allow variables (with optional indexes) on the left;"
                     << std::endl
                     << "    if you see an outer function logical_lt (<) with negated (-) second argument,"
                     << std::endl
                     << "    it indicates an assignment statement A <- B with illegal left"
                     << std::endl
                     << "    side A parsed as expression (A < (-B))."
                     << std::endl
                     << "  * Sampling statements allow arbitrary value-denoting expressions on the left."
                     << std::endl
                     << "  * Functions used as statements must be declared to have void returns"
                     << std::endl << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<expression_as_statement> expression_as_statement_f;

    struct unscope_locals {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(const std::vector<var_decl>& var_decls,
                      variable_map& vm) const {
        for (size_t i = 0; i < var_decls.size(); ++i)
          vm.remove(var_decls[i].name());
      }
    };
    boost::phoenix::function<unscope_locals> unscope_locals_f;

    struct add_while_condition {
      template <typename T1, typename T2, typename T3>
      struct result { typedef bool type; };
      bool operator()(while_statement& ws,
                      const expression& e,
                      std::stringstream& error_msgs) const {
        if (!e.expression_type().is_primitive()) {
          error_msgs << "conditions in while statement must be primitive int or real;"
                     << " found type=" << e.expression_type() << std::endl;
          return false;
        }
        ws.condition_ = e;
        return true;
      }               
    };
    boost::phoenix::function<add_while_condition> add_while_condition_f;

    struct add_while_body {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(while_statement& ws,
                      const statement& s) const {
        ws.body_ = s;
      }
    };
    boost::phoenix::function<add_while_body> add_while_body_f;
    
    struct add_loop_identifier {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };
      bool operator()(const std::string& name, 
                      std::string& name_local,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {
        name_local = name;
        if (vm.exists(name)) {
          error_msgs << "ERROR: loop variable already declared."
                     << " variable name=\"" << name << "\"" << std::endl;
          return false; // variable exists
        }
        vm.add(name, 
               base_var_decl(name,std::vector<expression>(),
                             INT_T),
               local_origin); // loop var acts like local
        return true;
      }
    };
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    struct remove_loop_identifier {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(const std::string& name, 
                      variable_map& vm) const {
        vm.remove(name);
      }
    };
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;

    struct validate_int_expr2 {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting integer required; found type=" 
                     << expr.expression_type() << std::endl;
          pass = false;
          return;
        }
        pass = true;
        return;
      }
    };
    boost::phoenix::function<validate_int_expr2> validate_int_expr2_f;

    struct validate_allow_sample {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(const bool& allow_sample,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!allow_sample) {
          error_msgs << "sampling only allowed in model."
                     << std::endl;
          pass = false;
          return;
        }
        pass = true;
        return;
      }
    };
    boost::phoenix::function<validate_allow_sample> validate_allow_sample_f;

    struct validate_non_void_expression {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      
      void operator()(const expression& e, 
                      bool& pass, 
                      std::ostream& error_msgs) const {
        pass = !e.expression_type().is_void();
        if (!pass) {
          error_msgs << "attempt to increment log prob with void expression" << std::endl;
        }
      }
    };
    boost::phoenix::function<validate_non_void_expression> validate_non_void_expression_f;
    

    template <typename Iterator>
    statement_grammar<Iterator>::statement_grammar(variable_map& var_map,
                                                   std::stringstream& error_msgs)
      : statement_grammar::base_type(statement_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map,error_msgs),
        var_decls_g(var_map,error_msgs),
        statement_2_g(var_map,error_msgs,*this)
    {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;

      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;
      using boost::spirit::qi::labels::_r3;

      // inherited features
      //   _r1 true if sample_r allowed
      //   _r2 source of variables allowed for assignments
      //   _r3 true if return_r allowed 
      statement_r.name("statement");
      statement_r
        %= no_op_statement_r                        // key ";"
        | statement_seq_r(_r1,_r2,_r3)              // key "{"
        | increment_log_prob_statement_r(_r1,_r2)   // key "increment_log_prob"
        | for_statement_r(_r1,_r2,_r3)              // key "for"
        | while_statement_r(_r1,_r2,_r3)            // key "while"
        | statement_2_g(_r1,_r2,_r3)                // key "if"
        | print_statement_r(_r2)                    // key "print"
        | reject_statement_r(_r2)                   // key "reject"
        | return_statement_r(_r2)                   // key "return"
        | void_return_statement_r(_r2)              // key "return"
        | assignment_r(_r2)                         // lvalue "<-"
        | sample_r(_r1,_r2)                         // expression "~"
        | expression_g(_r2)                         // expression
          [expression_as_statement_f(_pass,_1,boost::phoenix::ref(error_msgs_))]
        ;

      // _r1, _r2, _r3 same as statement_r
      statement_seq_r.name("sequence of statements");
      statement_seq_r
        %= lit('{')
        > local_var_decls_r[_a = _1]
        > *statement_r(_r1,_r2,_r3)
        > lit('}')
        > eps[unscope_locals_f(_a,boost::phoenix::ref(var_map_))]
        ;

      local_var_decls_r
        %= var_decls_g(false,local_origin); // - constants

      // inherited  _r1 = true if samples allowed as statements
      increment_log_prob_statement_r.name("increment log prob statement");
      increment_log_prob_statement_r
        %= lit("increment_log_prob") 
        > eps[ validate_allow_sample_f(_r1,_pass,
                                       boost::phoenix::ref(error_msgs_)) ]
        > lit('(')
        > expression_g(_r2) [ validate_non_void_expression_f(_1,_pass,
                                                             boost::phoenix::ref(error_msgs_)) ]
        > lit(')')
        > lit(';') 
        ;

      // _r1, _r2, _r3 same as statement_r
      while_statement_r.name("while statement");
      while_statement_r
        = lit("while")
        > lit('(')
        > expression_g(_r2)
          [_pass = add_while_condition_f(_val,_1,
                                         boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > statement_r(_r1,_r2,_r3)
          [add_while_body_f(_val,_1)]
        ;
      

      // _r1, _r2, _r3 same as statement_r
      for_statement_r.name("for statement");
      for_statement_r
        %= lit("for")
        > lit('(')
        > identifier_r [_pass 
                        = add_loop_identifier_f(_1,_a,
                                                boost::phoenix::ref(var_map_),
                                                boost::phoenix::ref(error_msgs_))]
        > lit("in")
        > range_r(_r2)
        > lit(')')
        > statement_r(_r1,_r2,_r3)
        > eps 
        [remove_loop_identifier_f(_a,boost::phoenix::ref(var_map_))];
      ;

      print_statement_r.name("print statement");
      print_statement_r
        %= lit("print")
        > lit('(')
        > (printable_r(_r1) % ',')
        > lit(')');

      // reject
      reject_statement_r.name("reject statement");
      reject_statement_r
        %= lit("reject")
        > lit('(')
        > (printable_r(_r1) % ',')
        > lit(')');

      printable_r.name("printable");
      printable_r
        %= printable_string_r 
        | expression_g(_r1);

      printable_string_r.name("printable quoted string");
      printable_string_r
        %= lit('"')
        > no_skip[*char_("a-zA-Z0-9/~!@#$%^&*()`_+-={}|[]:;'<>?,./ ")]
        > lit('"');
      
      identifier_r.name("identifier");
      identifier_r
        %= (lexeme[char_("a-zA-Z") 
                   >> *char_("a-zA-Z0-9_.")]);

      range_r.name("range expression pair, colon");
      range_r 
        %= expression_g(_r1)
           [validate_int_expr2_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        >> lit(':') 
        >> expression_g(_r1)
           [validate_int_expr2_f(_1,_pass,boost::phoenix::ref(error_msgs_))];

      assignment_r.name("variable assignment by expression");
      assignment_r
        %= ( var_lhs_r(_r1)
             >> lit("<-") )
        > expression_g(_r1)
        > lit(';')
          [_pass = validate_assignment_f(_val,_r1,boost::phoenix::ref(var_map_),
                                         boost::phoenix::ref(error_msgs_))]
        ;

      var_lhs_r.name("variable and array dimensions");
      var_lhs_r 
        %= identifier_r 
        >> opt_dims_r(_r1);

      opt_dims_r.name("array dimensions (optional)");
      opt_dims_r 
        %=  * dims_r(_r1);

      dims_r.name("array dimensions");
      dims_r 
        %= lit('[') 
        > (expression_g(_r1)
           [validate_int_expr2_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
           % ',')
        > lit(']')
        ;

      // inherited  _r1 = true if samples allowed as statements
      sample_r.name("distribution of expression");
      sample_r 
        %= ( expression_g(_r2)
             >> lit('~') )
        > eps
          [validate_allow_sample_f(_r1,_pass,
                                   boost::phoenix::ref(error_msgs_))]
        > distribution_r(_r2)
        > -truncation_range_r(_r2)
        > lit(';')
        > eps
          [_pass = validate_sample_f(_val,
                                     boost::phoenix::ref(var_map_),
                                     boost::phoenix::ref(error_msgs_))]
        ;

      distribution_r.name("distribution and parameters");
      distribution_r
        %= ( identifier_r
             >> lit('(')
             >> -(expression_g(_r1) % ',') )
        > lit(')')
        ;

      truncation_range_r.name("range pair");
      truncation_range_r
        %= lit('T')
        > lit('[') 
        > -expression_g(_r1)
        > lit(',')
        > -expression_g(_r1)
        > lit(']')
        ;

      // _r1 = allow sampling, _r2 = var origin
      return_statement_r.name("return statement");
      return_statement_r
        %= lit("return")
        >> expression_g(_r1)
        >> lit(';') [ validate_return_allowed_f(_r1,_pass,
                                                boost::phoenix::ref(error_msgs_)) ]
        ;

      // _r1 = var origin
      void_return_statement_r.name("void return statement");
      void_return_statement_r
        = lit("return")[_val = expression()]
        >> lit(';') [ validate_void_return_allowed_f(_r1,_pass,
                                                     boost::phoenix::ref(error_msgs_)) ]
        ;

      no_op_statement_r.name("no op statement");
      no_op_statement_r 
        %= lit(';') [_val = no_op_statement()];  // ok to re-use instance

      using boost::spirit::qi::on_error;
      using boost::spirit::qi::fail;
      using boost::spirit::qi::rethrow;
      using namespace boost::spirit::qi::labels;
      
    }

  }
}
#endif
