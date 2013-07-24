#ifndef __STAN__GM__PARSER__STATEMENT_GRAMMAR_DEF__HPP__
#define __STAN__GM__PARSER__STATEMENT_GRAMMAR_DEF__HPP__

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
// FIXME: get rid of unused include
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
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

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/common_adaptors_def.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::gm::assignment,
                          (stan::gm::variable_dims, var_dims_)
                          (stan::gm::expression, expr_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::variable_dims,
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::distribution,
                          (std::string, family_)
                          (std::vector<stan::gm::expression>, args_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::for_statement,
                          (std::string, variable_)
                          (stan::gm::range, range_)
                          (stan::gm::statement, statement_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::print_statement,
                          (std::vector<stan::gm::printable>, printables_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::sample,
                          (stan::gm::expression, expr_)
                          (stan::gm::distribution, dist_) 
                          (stan::gm::range, truncation_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::statements,
                          (std::vector<stan::gm::var_decl>, local_decl_)
                          (std::vector<stan::gm::statement>, statements_) )

namespace stan {

  namespace gm {

    struct validate_assignment {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };

      bool operator()(assignment& a,
                      const var_origin& origin_allowed,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {

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
                     << "; variable array dimensions = " << lhs_var_num_dims;
          return false;
        }
        if (lhs_type.num_dims_ != a.expr_.expression_type().num_dims_) {
          error_msgs << "mismatched dimensions on left- and right-hand side of assignment"
                     << "; left dims=" << lhs_type.num_dims_
                     << "; right dims=" << a.expr_.expression_type().num_dims_
                     << std::endl;
          return false;
        }

        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
        // int -> double promotion
        bool types_compatible 
          = lhs_base_type == rhs_base_type
          || ( lhs_base_type == DOUBLE_T && rhs_base_type == INT_T );
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; left variable=" << a.var_dims_.name_
                     << "; left base type=";
          write_base_expr_type(error_msgs,lhs_base_type);
          error_msgs << "; right base type=";
          write_base_expr_type(error_msgs,rhs_base_type);
          error_msgs << std::endl;
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
      bool operator()(const sample& s,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const {
        std::vector<expr_type> arg_types;
        arg_types.push_back(s.expr_.expression_type());
        for (size_t i = 0; i < s.dist_.args_.size(); ++i)
          arg_types.push_back(s.dist_.args_[i].expression_type());
        std::string function_name(s.dist_.family_);
        function_name += "_log";
        // expr_type result_type 
        // = function_signatures::instance()
        // .get_result_type(function_name,arg_types,error_msgs);
        // if (!result_type.is_primitive_double()) {
        if (!is_double_return(function_name,arg_types,error_msgs)) {
          error_msgs << "unknown distribution=" << s.dist_.family_ << std::endl;
          return false;
        }
        // test for LHS not being purely a variable
        if (has_non_param_var(s.expr_,var_map)) {
          // FIXME:  really want to get line numbers in here too
          error_msgs << "Warning (non-fatal):"
                     << "     sampling statement (~) contains a transformed parameter or local variable."
                     << std::endl
                     << "     You must increment lp__ with the log absolute determinant"
                     << " of the Jacobian of the transform."
                     << std::endl
                     << "     Sampling Statement left-hand-side expression:"
                     << std::endl
                     << "          ";
          generate_expression(s.expr_,error_msgs);
          error_msgs << " ~ ";
          error_msgs << function_name << "(...)";
          error_msgs << std::endl;
        }
        if (s.truncation_.has_low()) {
          std::vector<expr_type> arg_types_trunc(arg_types);
          arg_types_trunc[0] = s.truncation_.low_.expression_type(); 
          std::string function_name_cdf(s.dist_.family_);
          function_name_cdf += "_cdf";
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
          function_name_cdf += "_cdf";
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

    // struct add_conditional_condition {
    //   template <typename T1, typename T2, typename T3>
    //   struct result { typedef bool type; };
    //   bool operator()(conditional_statement& cs,
    //                   const expression& e,
    //                   std::stringstream& error_msgs) const {
    //     if (!e.expression_type().is_primitive()) {
    //       error_msgs << "conditions in if-else statement must be primitive int or real;"
    //                  << " found type=" << e.expression_type() << std::endl;
    //       return false;
    //     }
    //     cs.conditions_.push_back(e);
    //     return true;
    //   }               
    // };
    // boost::phoenix::function<add_conditional_condition> add_conditional_condition_f;

    // struct add_conditional_body {
    //   template <typename T1, typename T2>
    //   struct result { typedef void type; };
    //   void operator()(conditional_statement& cs,
    //                   const statement& s) const {
    //     cs.bodies_.push_back(s);
    //   }
    // };
    // boost::phoenix::function<add_conditional_body> add_conditional_body_f;

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
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const expression& expr,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting integer required; found type=" 
                     << expr.expression_type() << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_int_expr2> validate_int_expr2_f;

    struct validate_allow_sample {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const bool& allow_sample,
                      std::stringstream& error_msgs) const {
        if (!allow_sample) {
          error_msgs << "ERROR:  sampling only allowed in model."
                     << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_allow_sample> validate_allow_sample_f;


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

      // _r1 true if sample_r allowed (inherited)
      // _r2 source of variables allowed for assignments
      // set to true if sample_r are allowed
      statement_r.name("statement");
      statement_r
        %= statement_seq_r(_r1,_r2)
        | for_statement_r(_r1,_r2)
        | while_statement_r(_r1,_r2)
        | statement_2_g(_r1,_r2)
        | print_statement_r(_r2)
        | assignment_r(_r2)
          [_pass 
            = validate_assignment_f(_1,_r2,boost::phoenix::ref(var_map_),
                                     boost::phoenix::ref(error_msgs_))]
        | sample_r(_r1,_r2) [_pass = validate_sample_f(_1,
                                               boost::phoenix::ref(var_map_),
                                               boost::phoenix::ref(error_msgs_))]
        | no_op_statement_r
        ;

      // _r1, _r2 same as statement_r
      statement_seq_r.name("sequence of statements");
      statement_seq_r
        %= lit('{')
        > local_var_decls_r[_a = _1]
        > *statement_r(_r1,_r2)
        > lit('}')
        > eps[unscope_locals_f(_a,boost::phoenix::ref(var_map_))]
        ;

      local_var_decls_r
        %= var_decls_g(false,local_origin); // - constants

      while_statement_r.name("while statement");
      while_statement_r
        = lit("while")
        > lit('(')
        > expression_g(_r2)
          [_pass = add_while_condition_f(_val,_1,
                                         boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > statement_r(_r1,_r2)
          [add_while_body_f(_val,_1)]
        ;
      

      // _r1, _r2 same as statement_r
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
        > statement_r(_r1,_r2)
        > eps 
        [remove_loop_identifier_f(_a,boost::phoenix::ref(var_map_))];
      ;

      print_statement_r.name("print statement");
      print_statement_r
        %= lit("print")
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
        [_pass = validate_int_expr2_f(_1,boost::phoenix::ref(error_msgs_))]
        >> lit(':') 
        >> expression_g(_r1)
        [_pass = validate_int_expr2_f(_1,boost::phoenix::ref(error_msgs_))];

      assignment_r.name("variable assignment by expression");
      assignment_r
        %= ( var_lhs_r(_r1)
             >> lit("<-") )
        > expression_g(_r1)
        > lit(';') 
        ;

      var_lhs_r.name("variable and array dimensions");
      var_lhs_r 
        %= identifier_r 
        >> opt_dims_r(_r1);

      opt_dims_r.name("array dimensions (optional)");
      opt_dims_r 
        %=  - dims_r(_r1);

      dims_r.name("array dimensions");
      dims_r 
        %= lit('[') 
        > (expression_g(_r1)
           [_pass = validate_int_expr2_f(_1,boost::phoenix::ref(error_msgs_))]
           % ',')
        > lit(']')
        ;

      // inherited  _r1 = true if samples allowed as statements
      sample_r.name("distribution of expression");
      sample_r 
        %= ( expression_g(_r2)
             >> lit('~') )
        > eps
        [_pass 
         = validate_allow_sample_f(_r1,boost::phoenix::ref(error_msgs_))] 
        > distribution_r(_r2)
        > -truncation_range_r(_r2)
        > lit(';');

      distribution_r.name("distribution and parameters");
      distribution_r
        %= ( identifier_r
             >> lit('(')
             >> -(expression_g(_r1) % ',') )
        > lit(')');

      truncation_range_r.name("range pair");
      truncation_range_r
        %= lit('T')
        > lit('[') 
        > -expression_g(_r1)
        > lit(',')
        > -expression_g(_r1)
        > lit(']');

      no_op_statement_r.name("no op statement");
      no_op_statement_r 
        %= lit(';') [_val = no_op_statement()];  // ok to re-use instance

    }

  }
}
#endif
