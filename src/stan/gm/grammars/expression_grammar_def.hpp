#ifndef STAN__GM__PARSER__EXPRESSION_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__EXPRESSION_GRAMMAR_DEF__HPP

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


    // FIXME: cut and paste from term grammar, having trouble w. includes
    struct set_fun_type2 {
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
    boost::phoenix::function<set_fun_type2> set_fun_type2_f;

   struct binary_op_expr {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef expression type; };

      expression operator()(expression& expr1,
                            const expression& expr2,
                            const std::string& op,
                            const std::string& fun_name,
                            std::ostream& error_msgs) const {
        if (!expr1.expression_type().is_primitive()
            || !expr2.expression_type().is_primitive()) {
          error_msgs << "binary infix operator "
                     << op 
                     << " with functional interpretation "
                     << fun_name
                     << " requires arguments or primitive type (int or real)"
                     << ", found left type=" << expr1.expression_type()
                     << ", right arg type=" << expr2.expression_type()
                     << "; ";
        }
        std::vector<expression> args;
        args.push_back(expr1);
        args.push_back(expr2);
        set_fun_type2 sft;
        fun f(fun_name,args);
        sft(f,error_msgs);
        return expression(f);
      }
    };
    boost::phoenix::function<binary_op_expr> binary_op_f;


    template <typename Iterator>
    expression_grammar<Iterator>::expression_grammar(variable_map& var_map,
                                                     std::stringstream& error_msgs)
      : expression_grammar::base_type(expression_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression07_g(var_map,error_msgs,*this)
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

      expression_r.name("expression (top level, precedence 15, binary ||");
      expression_r
        = expression14_r(_r1) [_val = _1]
        > *( lit("||") 
             > expression14_r(_r1)  [_val = binary_op_f(_val,_1,"||","logical_or",
                                                   boost::phoenix::ref(error_msgs))] 
             );

      expression14_r.name("expression, precedence 14, binary &&");
      expression14_r 
        = expression10_r(_r1) [_val = _1]
        > *( lit("&&") 
             > expression10_r(_r1)  [_val = binary_op_f(_val,_1,"&&","logical_and",
                                                   boost::phoenix::ref(error_msgs))] 
             );

      expression10_r.name("expression, precedence 10, binary ==, !=");
      expression10_r 
        = expression09_r(_r1) [_val = _1]
        > *( ( lit("==") 
               > expression09_r(_r1)  [_val = binary_op_f(_val,_1,"==","logical_eq",
                                                       boost::phoenix::ref(error_msgs))] )
              |
              ( lit("!=") 
                > expression09_r(_r1)  [_val = binary_op_f(_val,_1,"!=","logical_neq",
                                                      boost::phoenix::ref(error_msgs))] ) 
              );

      expression09_r.name("expression, precedence 9, binary <, <=, >, >=");
      expression09_r 
        = expression07_g(_r1) [_val = _1]
        > *( ( lit("<=")
               > expression07_g(_r1)  [_val = binary_op_f(_val,_1,"<","logical_lte",
                                                      boost::phoenix::ref(error_msgs))] )
              |
              ( lit("<") 
                > expression07_g(_r1)  [_val = binary_op_f(_val,_1,"<=","logical_lt",
                                                      boost::phoenix::ref(error_msgs))] ) 
              |
              ( lit(">=") 
                > expression07_g(_r1)  [_val = binary_op_f(_val,_1,">","logical_gte",
                                                      boost::phoenix::ref(error_msgs))] ) 
              |
              ( lit(">") 
                > expression07_g(_r1)  [_val = binary_op_f(_val,_1,">=","logical_gt",
                                                      boost::phoenix::ref(error_msgs))] ) 
              );

    }
  }
}

#endif
