#ifndef STAN__GM__PARSER__EXPRESSION_GRAMMAR07_DEF__HPP__
#define STAN__GM__PARSER__EXPRESSION_GRAMMAR07_DEF__HPP__

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
#include <stan/gm/grammars/expression07_grammar.hpp>


namespace stan { 

  namespace gm {

    struct validate_expr_type3 {
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
    boost::phoenix::function<validate_expr_type3> validate_expr_type3_f;

    // FIXME: cut and paste from term grammar, having trouble w. includes
    struct set_fun_type3 {
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
    boost::phoenix::function<set_fun_type3> set_fun_type3_f;

    struct addition_expr3 {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {
        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 += expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type3 sft;
        fun f("add",args);
        sft(f,error_msgs);
        return expression(f);
        return expr1 += expr2;
      }
    };
    boost::phoenix::function<addition_expr3> addition3_f;


    struct subtraction_expr3 {
      template <typename T1, typename T2, typename T3>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            std::ostream& error_msgs) const {
        if (expr1.expression_type().is_primitive()
            && expr2.expression_type().is_primitive()) {
          return expr1 -= expr2;
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type3 sft;
        fun f("subtract",args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<subtraction_expr3> subtraction3_f;



    template <typename Iterator>
    expression07_grammar<Iterator>::expression07_grammar(variable_map& var_map,
                                                         std::stringstream& error_msgs,
                                                         expression_grammar<Iterator>& eg)
      : expression07_grammar::base_type(expression07_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        term_g(var_map,error_msgs,eg)
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
      
      expression07_r.name("expression, precedence 7, binary +, -");
      expression07_r 
        =  term_g(_r1)
            [_val = _1]
        > *( ( lit('+')
               > term_g(_r1) // expression07_r       
                [_val = addition3_f(_val,_1,boost::phoenix::ref(error_msgs))] )
              |  
              ( lit('-') 
                > term_g(_r1) // expression07_r   
                [_val = subtraction3_f(_val,_1,boost::phoenix::ref(error_msgs))] )
              )
        > eps[_pass = validate_expr_type3_f(_val,boost::phoenix::ref(error_msgs_))]
        ;


    }
  }
}

#endif
