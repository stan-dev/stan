#ifndef STAN__GM__PARSER__TERM_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__TERM_GRAMMAR_DEF__HPP

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
#include <stan/gm/grammars/term_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::gm::index_op,
                          (stan::gm::expression, expr_)
                          (std::vector<std::vector<stan::gm::expression> >, 
                           dimss_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::integrate_ode,
                          (std::string, system_function_name_)
                          (stan::gm::expression, y0_)
                          (stan::gm::expression, t0_)
                          (stan::gm::expression, ts_)
                          (stan::gm::expression, theta_)
                          (stan::gm::expression, x_)
                          (stan::gm::expression, x_int_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::fun,
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, args_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::int_literal,
                          (int,val_)
                          (stan::gm::expr_type,type_));

BOOST_FUSION_ADAPT_STRUCT(stan::gm::double_literal,
                          (double,val_)
                          (stan::gm::expr_type,type_) );


namespace stan { 

  namespace gm {


    struct validate_integrate_ode {

      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };

      void operator()(const integrate_ode& ode_fun,
                      const variable_map& var_map,
                      bool& pass,
                      std::ostream& error_msgs) const {
        pass = true;

        // test function argument type
        expr_type sys_result_type(DOUBLE_T,1);
        std::vector<expr_type> sys_arg_types;
        sys_arg_types.push_back(expr_type(DOUBLE_T,0));
        sys_arg_types.push_back(expr_type(DOUBLE_T,1));
        sys_arg_types.push_back(expr_type(DOUBLE_T,1));
        sys_arg_types.push_back(expr_type(DOUBLE_T,1));
        sys_arg_types.push_back(expr_type(INT_T,1));
        function_signature_t system_signature(sys_result_type, sys_arg_types);
        if (!function_signatures::instance()
            .is_defined(ode_fun.system_function_name_,system_signature)) {
          error_msgs << "first argument to integrate_ode must be a function with signature"
                     << " (real, real[], real[], real[], int[]) : real[] ";
          pass = false;
        }

        // test regular argument types
        if (ode_fun.y0_.expression_type() != expr_type(DOUBLE_T,1)) {
          error_msgs << "second argument to integrate_ode must be type real[]"
                     << " for intial system state"
                     << "; found type=" 
                     << ode_fun.y0_.expression_type()
                     << ". ";
          pass = false;
        } 
        if (!ode_fun.t0_.expression_type().is_primitive()) {
          error_msgs << "third argument to integrate_ode must be type real or int"
                     << " for initial time"
                     << "; found type=" 
                     << ode_fun.t0_.expression_type()
                     << ". ";
          pass = false;
        }
        if (ode_fun.ts_.expression_type() != expr_type(DOUBLE_T,1)) {
          error_msgs << "fourth argument to integrate_ode must be type real[]"
                     << " for requested solution times"
                     << "; found type=" 
                     << ode_fun.ts_.expression_type()
                     << ". ";
          pass = false;
        }
        if (ode_fun.theta_.expression_type() != expr_type(DOUBLE_T,1)) {
          error_msgs << "fifth argument to integrate_ode must be type real[]"
                     << " for parameters"
                     << "; found type=" 
                     << ode_fun.theta_.expression_type()
                     << ". ";
          pass = false;
        }
        if (ode_fun.x_.expression_type() != expr_type(DOUBLE_T,1)) {
          error_msgs << "sixth argument to integrate_ode must be type real[]"
                     << " for real data;"
                     << " found type=" 
                     << ode_fun.x_.expression_type()
                     << ". ";
          pass = false;
        }
        if (ode_fun.x_int_.expression_type() != expr_type(INT_T,1)) {
          error_msgs << "seventh argument to integrate_ode must be type int[]"
                     << " for integer data;"
                     << " found type=" 
                     << ode_fun.x_int_.expression_type()
                     << ". ";
          pass = false;
        }

        // test data-only variables do not have parameters
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
    };
    boost::phoenix::function<validate_integrate_ode> validate_integrate_ode_f;
    
    struct set_fun_type {
      template <typename T1, typename T2>
      struct result { typedef fun type; };

      fun operator()(fun& fun,
                     std::ostream& error_msgs) const {
        std::vector<expr_type> arg_types;
        for (size_t i = 0; i < fun.args_.size(); ++i)
          arg_types.push_back(fun.args_[i].expression_type());
        fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                    arg_types,
                                                                    error_msgs);
        return fun;
      }
    };
    boost::phoenix::function<set_fun_type> set_fun_type_f;


    struct set_fun_type_named {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef void type; };

      void operator()(expression& fun_result,
                      fun& fun,
                      const var_origin& var_origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        std::vector<expr_type> arg_types;
        for (size_t i = 0; i < fun.args_.size(); ++i)
          arg_types.push_back(fun.args_[i].expression_type());
        fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                    arg_types,
                                                                    error_msgs);


        if (has_rng_suffix(fun.name_)) {  
          if (!( var_origin == derived_origin
                 || var_origin == function_argument_origin_rng )) {
            error_msgs << "random number generators only allowed in"
                       << " generated quantities block or"
                       << " user-defined functions with names ending in _rng"
                       << "; found function=" << fun.name_
                       << " in block=";
          print_var_origin(error_msgs,var_origin);
          error_msgs << std::endl;
          pass = false;
          return;
          }
        }

        if (has_lp_suffix(fun.name_)) {
          // modified function_argument_origin to add _lp because
          // that's only viable context
          if (!( var_origin == parameter_origin
                 || var_origin == transformed_parameter_origin
                 || var_origin == function_argument_origin_lp 
                 || var_origin == local_origin )) {
            error_msgs << "lp suffixed functions only allowed in"
                       << " transformed parameter, function argument, or model"

                       << " in block=";
            print_var_origin(error_msgs,var_origin);
            error_msgs << std::endl;
            pass = false;
            return;
          }
        }

        if (fun.name_ == "abs" 
            && fun.args_.size() > 0 
            && fun.args_[0].expression_type().is_primitive_double()) {
          error_msgs << "Warning: Function abs(real) is deprecated."
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
    };
    boost::phoenix::function<set_fun_type_named> set_fun_type_named_f;
 
    struct exponentiation_expr {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef void type; };

      void operator()(expression& expr1,
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
          print_var_origin(error_msgs,var_origin);
          error_msgs << std::endl;
          pass = false;
          return;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        fun f("pow",args);
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<exponentiation_expr> exponentiation_f;

    struct multiplication_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr1,
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
        fun f("multiply",args);
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<multiplication_expr> multiplication_f;

    void generate_expression(const expression& e, std::ostream& o);

    struct division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr1,
                      const expression& expr2,
                      std::ostream& error_msgs) const {
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
          error_msgs << "Warning: integer division implicitly rounds to integer."
                     << " Found int division: ";
          generate_expression(expr1.expr_,error_msgs);
          error_msgs << " / ";
          generate_expression(expr2.expr_,error_msgs);
          error_msgs << std::endl
                     << " Positive values rounded down, negative values rounded up or down"
                     << " in platform-dependent way."
                     << std::endl;

          fun f("divide",args);
          sft(f,error_msgs);
          expr1 = expression(f);
          return;
        }
        if ((expr1.expression_type().type() == MATRIX_T
             || expr1.expression_type().type() == ROW_VECTOR_T)
            && expr2.expression_type().type() == MATRIX_T) {
          fun f("mdivide_right",args);
          sft(f,error_msgs);
          expr1 = expression(f);
          return;
        }
        fun f("divide",args);
        sft(f,error_msgs);
        expr1 = expression(f);
        return;
      }
    };
    boost::phoenix::function<division_expr> division_f;

    struct modulus_expr {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };

      void operator()(expression& expr1,
                      const expression& expr2,
                      bool& pass,
                      std::ostream& error_msgs) const {
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
        fun f("modulus",args);
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<modulus_expr> modulus_f;

    struct left_division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr1,
                      const expression& expr2,
                      std::ostream& error_msgs) const {
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        if (expr1.expression_type().type() == MATRIX_T
            && (expr2.expression_type().type() == VECTOR_T
                || expr2.expression_type().type() == MATRIX_T)) {
          fun f("mdivide_left",args);
          sft(f,error_msgs);
          expr1 = expression(f);
          return;
        }
        fun f("divide_left",args); // this doesn't exist, so will
                                   // throw error on purpose
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<left_division_expr> left_division_f;

    struct elt_multiplication_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr1,
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
        fun f("elt_multiply",args);
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<elt_multiplication_expr> elt_multiplication_f;

    struct elt_division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr1,
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
        fun f("elt_divide",args);
        sft(f,error_msgs);
        expr1 = expression(f);
      }
    };
    boost::phoenix::function<elt_division_expr> elt_division_f;

    // Cut-and-Paste from Spirit examples, including comment:  We
    // should be using expression::operator-. There's a bug in phoenix
    // type deduction mechanism that prevents us from doing
    // so. Phoenix will be switching to BOOST_TYPEOF. In the meantime,
    // we will use a phoenix::function below:
    struct negate_expr {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };

      void operator()(expression& expr_result,
                      const expression& expr,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (expr.expression_type().is_primitive()) {
          expr_result = expression(unary_op('-', expr));
          return;
        }
        std::vector<expression> args;
        args.push_back(expr);
        set_fun_type sft;
        fun f("minus",args);
        sft(f,error_msgs);
        expr_result = expression(f);
      }
    };
    boost::phoenix::function<negate_expr> negate_expr_f;

    struct logical_negate_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(expression& expr_result,
                      const expression& expr,
                      std::ostream& error_msgs) const {
        if (!expr.expression_type().is_primitive()) {
          error_msgs << "logical negation operator ! only applies to int or real types; ";
          expr_result = expression();
        }
        std::vector<expression> args;
        args.push_back(expr);
        set_fun_type sft;
        fun f("logical_negation",args);
        sft(f,error_msgs);
        expr_result = expression(f);
      }
    };
    boost::phoenix::function<logical_negate_expr> logical_negate_expr_f;

    struct transpose_expr {
      template <typename T1, typename T2>
      struct result { typedef expression type; };

      expression operator()(const expression& expr,
                            std::ostream& error_msgs) const {

        if (expr.expression_type().is_primitive()) {
          return expr; // transpose of basic is self -- works?
        }
        std::vector<expression> args;
        args.push_back(expr);
        set_fun_type sft;
        fun f("transpose",args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<transpose_expr> transpose_f;

    struct add_expression_dimss {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(expression& expression,
                      std::vector<std::vector<stan::gm::expression> >& dimss,
                      bool& pass,
                      std::ostream& error_msgs) const {
        index_op iop(expression,dimss);
        iop.infer_type();
        if (iop.type_.is_ill_formed()) {
          error_msgs << "indexes inappropriate for expression." << std::endl;
          pass = false;
        } 
        pass = true;
        expression = iop;
      }
    };
    boost::phoenix::function<add_expression_dimss> add_expression_dimss_f;

    struct set_var_type {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef variable type; };
      variable operator()(variable& var_expr, 
                          variable_map& vm,
                          std::ostream& error_msgs,
                          bool& pass) const {
        std::string name = var_expr.name_;
        if (!vm.exists(name)) {
          pass = false;
          error_msgs << "variable \"" << name << '"' << " does not exist." 
                     << std::endl;
          return var_expr;
        }
        if (name == std::string("lp__")) {
          error_msgs << std::endl
                     << "WARNING:"
                     << std::endl
                     << "  Direct use of variable lp__ is deprecated and will be removed in a future release."
                     << std::endl
                     << "  Please use increment_log_prob(u) in place of of lp__ <- lp__ + u."
                     << std::endl;
        }
        pass = true;
        var_expr.set_type(vm.get_base_type(name),vm.get_num_dims(name));
        return var_expr;
      }
    };
    boost::phoenix::function<set_var_type> set_var_type_f;

    struct validate_int_expr3 {
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
    boost::phoenix::function<validate_int_expr3> validate_int_expr3_f;



    template <typename Iterator>
    term_grammar<Iterator>::term_grammar(variable_map& var_map,
                                         std::stringstream& error_msgs,
                                         expression_grammar<Iterator>& eg)
      : term_grammar::base_type(term_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(eg)
    {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::double_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::int_;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;
      using boost::spirit::qi::labels::_r1;

      // _r1 : var_origin

      term_r.name("term");
      term_r 
        = ( negated_factor_r(_r1)                       
            [_val = _1]
            >> *( (lit('*') > negated_factor_r(_r1)     
                              [multiplication_f(_val,_1,
                                                boost::phoenix::ref(error_msgs_))])
                  | (lit('/') > negated_factor_r(_r1)   
                                [division_f(_val,_1,
                                            boost::phoenix::ref(error_msgs_))])
                  | (lit('%') > negated_factor_r(_r1)   
                                [modulus_f(_val,_1,_pass,
                                           boost::phoenix::ref(error_msgs_))])
                  | (lit('\\') > negated_factor_r(_r1)   
                                 [left_division_f(_val,_1,
                                                   boost::phoenix::ref(error_msgs_))])
                  | (lit(".*") > negated_factor_r(_r1)   
                                 [elt_multiplication_f(_val,_1,
                                                        boost::phoenix::ref(error_msgs_))])
                  | (lit("./") > negated_factor_r(_r1)   
                                 [elt_division_f(_val,_1,
                                                 boost::phoenix::ref(error_msgs_))])
                   )
             )
        ;

      negated_factor_r 
        = lit('-') >> negated_factor_r(_r1) 
        [negate_expr_f(_val,_1,_pass,boost::phoenix::ref(error_msgs_))]
        | lit('!') >> negated_factor_r(_r1) 
                      [logical_negate_expr_f(_val,_1,boost::phoenix::ref(error_msgs_))]
        | lit('+') >> negated_factor_r(_r1)  [_val = _1]
        | exponentiated_factor_r(_r1) [_val = _1]
        | indexed_factor_r(_r1) [_val = _1];


      exponentiated_factor_r.name("(optionally) exponentiated factor");
      exponentiated_factor_r 
        = ( indexed_factor_r(_r1) [_val = _1] 
            >> lit('^') 
            > negated_factor_r(_r1)
            [exponentiation_f(_val,_1,_r1,_pass,
                              boost::phoenix::ref(error_msgs_))] 
            )
        ;

      indexed_factor_r.name("(optionally) indexed factor [sub]");
      indexed_factor_r 
        = factor_r(_r1) [_val = _1]
        > * (  
             (+dims_r(_r1)) 
               [add_expression_dimss_f(_val, _1, _pass,
                                       boost::phoenix::ref(error_msgs_))]
               | 
               lit("'") 
               [_val = transpose_f(_val, boost::phoenix::ref(error_msgs_))] 
               )
        ;
      
      integrate_ode_r.name("solve ode");
      integrate_ode_r 
        %= lit("integrate_ode")
        > lit('(')
        > identifier_r          // system function name (function only)
        > lit(',')
        > expression_g(_r1)     // y0
        > lit(',')
        > expression_g(_r1)     // t0 (data only)
        > lit(',')
        > expression_g(_r1)     // ts (data only)
        > lit(',')
        > expression_g(_r1)     // theta
        > lit(',')
        > expression_g(_r1)     // x (data only)
        > lit(',')
        > expression_g(_r1)     // x_int (data only)
        > lit(')') [validate_integrate_ode_f(_val, 
                                         boost::phoenix::ref(var_map_),
                                         _pass,
                                         boost::phoenix::ref(error_msgs_))];

      factor_r.name("factor");
      factor_r =
        integrate_ode_r(_r1)    [_val = _1]
        | 
        fun_r(_r1)          [set_fun_type_named_f(_val,_1,_r1,_pass,
                                                    boost::phoenix::ref(error_msgs_))]
        | variable_r          [_val = set_var_type_f(_1,boost::phoenix::ref(var_map_),
                                                     boost::phoenix::ref(error_msgs_),
                                                     _pass)]
        | int_literal_r       [_val = _1]
        | double_literal_r    [_val = _1]
        | ( lit('(') 
            > expression_g(_r1)    [_val = _1]
            > lit(')') )
        ;

      int_literal_r.name("integer literal");
      int_literal_r
        %= int_ 
        >> !( lit('.')
              | lit('e')
              | lit('E') );


      double_literal_r.name("real literal");
      double_literal_r
        %= double_;


      fun_r.name("function and argument expressions");
      fun_r 
        %= identifier_r // no test yet on valid naming
        >> args_r(_r1);


      identifier_r.name("identifier (expression grammar)");
      identifier_r
        %= lexeme[char_("a-zA-Z") 
                  >> *char_("a-zA-Z0-9_.")];


      args_r.name("function argument expressions");
      args_r 
        %= (lit('(') >> lit(')'))
        | ( ( lit('(')
              >> (expression_g(_r1) % ',') )
            > lit(')') )
        ;

      
      dims_r.name("array dimensions");
      dims_r 
        %= lit('[') 
        > (expression_g(_r1)
           [_pass = validate_int_expr3_f(_1,boost::phoenix::ref(error_msgs_))]
           % ',')
        > lit(']')
        ;

 
      variable_r.name("variable expression");
      variable_r
        %= identifier_r 
        > !lit('(');    // negative lookahead to prevent failure in
                        // fun to try to evaluate as variable [cleaner
                        // error msgs]

    }
  }
}

#endif
