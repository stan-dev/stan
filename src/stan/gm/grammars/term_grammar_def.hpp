#ifndef __STAN__GM__PARSER__TERM_GRAMMAR_DEF__HPP__
#define __STAN__GM__PARSER__TERM_GRAMMAR_DEF__HPP__

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
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef fun type; };

      fun operator()(fun& fun,
                     const var_origin& var_origin,
                     bool& pass,
                     std::ostream& error_msgs) const {
        std::vector<expr_type> arg_types;
        for (size_t i = 0; i < fun.args_.size(); ++i)
          arg_types.push_back(fun.args_[i].expression_type());
        fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                    arg_types,
                                                                    error_msgs);

        pass = !has_rng_suffix(fun.name_) || var_origin == derived_origin;
        if (fun.name_ == "abs"
            && fun.args_[0].expression_type().is_primitive_double()) {
          error_msgs << "Warning: Function abs(real) is deprecated."
                     << std::endl
                     << "         It will be removed in a future release."
                     << std::endl
                     << "         Use fabs(real) instead."
                     << std::endl << std::endl;
        }


        if (!pass) {
          error_msgs << "random number generators only allowed in generated quantities block"
                     << "; found function=" << fun.name_
                     << " in block=";
          print_var_origin(error_msgs,var_origin);
          error_msgs << std::endl;
        }

        return fun;
      }
    };
    boost::phoenix::function<set_fun_type_named> set_fun_type_named_f;


 
    struct multiplication_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {

        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 *= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        fun f("multiply",args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<multiplication_expr> multiplication;

    void generate_expression(const expression& e, std::ostream& o);

    struct division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {
        if (expr1.expression_type().is_primitive_int() 
            && expr2.expression_type().is_primitive_int()) {
          // getting here, but not printing?  only print error if problems?
          error_msgs << "Warning: integer division implicitly rounds to integer."
                     << " Found int division: ";
          generate_expression(expr1.expr_,error_msgs);
          error_msgs << " / ";
          generate_expression(expr2.expr_,error_msgs);
          error_msgs << std::endl
                     << " Positive values rounded down, negative values rounded up or down"
                     << " in platform-dependent way."
                     << std::endl;
        }
            
        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 /= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        if ((expr1.expression_type().type() == MATRIX_T
             || expr1.expression_type().type() == ROW_VECTOR_T)
            && expr2.expression_type().type() == MATRIX_T) {
          fun f("mdivide_right",args);
          sft(f,error_msgs);
          return expression(f);
        }
        
        fun f("divide",args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<division_expr> division;

    struct left_division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {
        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 /= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        if (expr1.expression_type().type() == MATRIX_T
            && (expr2.expression_type().type() == VECTOR_T
                || expr2.expression_type().type() == MATRIX_T)) {
          fun f("mdivide_left",args);
          sft(f,error_msgs);
          return expression(f);
        }
        fun f("divide_left",args); // this doesn't exist, so will
                                   // throw error on purpose
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<left_division_expr> left_division;

    struct elt_multiplication_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {

        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 *= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        fun f("elt_multiply",args);
        sft(f,error_msgs);
        return expression(f);
        return expr1 += expr2;
      }
    };
    boost::phoenix::function<elt_multiplication_expr> elt_multiplication;

    struct elt_division_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {

        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 /= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type sft;
        fun f("elt_divide",args);
        sft(f,error_msgs);
        return expression(f);
        return expr1 += expr2;
      }
    };
    boost::phoenix::function<elt_division_expr> elt_division;

    // Cut-and-Paste from Spirit examples, including comment:  We
    // should be using expression::operator-. There's a bug in phoenix
    // type deduction mechanism that prevents us from doing
    // so. Phoenix will be switching to BOOST_TYPEOF. In the meantime,
    // we will use a phoenix::function below:
    struct negate_expr {
      template <typename T1, typename T2>
      struct result { typedef expression type; };

      expression operator()(const expression& expr,
                            std::ostream& error_msgs) const {
        if (expr.expression_type().is_primitive()) {
          return expression(unary_op('-', expr));
        }
        std::vector<expression> args;
        args.push_back(expr);
        set_fun_type sft;
        fun f("minus",args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<negate_expr> negate_expr_f;

    struct logical_negate_expr {
      template <typename T1, typename T2>
      struct result { typedef expression type; };

      expression operator()(const expression& expr,
                            std::ostream& error_msgs) const {
        if (!expr.expression_type().is_primitive()) {
          error_msgs << "logical negation operator ! only applies to int or real types; ";
          return expression();
        }
        std::vector<expression> args;
        args.push_back(expr);
        set_fun_type sft;
        fun f("logical_negation",args);
        sft(f,error_msgs);
        return expression(f);
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
      struct result { typedef T1 type; };
      expression operator()(expression& expression,
                            std::vector<std::vector<stan::gm::expression> >& dimss,
                            bool& pass,
                            std::ostream& error_msgs) const {
        index_op iop(expression,dimss);
        iop.infer_type();
        if (iop.type_.is_ill_formed()) {
          error_msgs << "indexes inappropriate for expression." << std::endl;
          pass = false;
        } else {
          pass = true;
        }
        return iop;
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


    struct validate_expr_type {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const expression& expr,
                      std::ostream& error_msgs) const {
        if (expr.expression_type().is_ill_formed()) {
          error_msgs << "expression is ill formed" << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_expr_type> validate_expr_type_f;

    



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
                               [_val = multiplication(_val,_1,
                                                      boost::phoenix::ref(error_msgs_))])
                  | (lit('/') > negated_factor_r(_r1)   
                                 [_val = division(_val,_1,boost::phoenix::ref(error_msgs_))])
                  | (lit('\\') > negated_factor_r(_r1)   
                                  [_val = left_division(_val,_1,
                                                        boost::phoenix::ref(error_msgs_))])
                  | (lit(".*") > negated_factor_r(_r1)   
                                  [_val = elt_multiplication(_val,_1,
                                                         boost::phoenix::ref(error_msgs_))])
                  | (lit("./") > negated_factor_r(_r1)   
                                  [_val = elt_division(_val,_1,
                                                       boost::phoenix::ref(error_msgs_))])
                   )
             )
        ;


      negated_factor_r 
        = lit('-') >> negated_factor_r(_r1) 
                      [_val = negate_expr_f(_1,boost::phoenix::ref(error_msgs_))]
        | lit('!') >> negated_factor_r(_r1) 
                      [_val = logical_negate_expr_f(_1,boost::phoenix::ref(error_msgs_))]
        | lit('+') >> negated_factor_r(_r1)  [_val = _1]
        | indexed_factor_r(_r1) [_val = _1];


      indexed_factor_r.name("(optionally) indexed factor [sub]");
      indexed_factor_r 
        = factor_r(_r1) [_val = _1]
        > * (  
             (+dims_r(_r1)) 
               [_val = add_expression_dimss_f(_val, _1, _pass,
                                            boost::phoenix::ref(error_msgs_))]
               | 
               lit("'") 
               [_val = transpose_f(_val, boost::phoenix::ref(error_msgs_))] 
               )
        ;


      factor_r.name("factor");
      factor_r
        =  int_literal_r     [_val = _1]
        | double_literal_r    [_val = _1]
        | fun_r(_r1)          [_val = set_fun_type_named_f(_1,_r1,_pass,boost::phoenix::ref(error_msgs_))]
        | variable_r          [_val = set_var_type_f(_1,boost::phoenix::ref(var_map_),
                                                     boost::phoenix::ref(error_msgs_),
                                                     _pass)]
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
