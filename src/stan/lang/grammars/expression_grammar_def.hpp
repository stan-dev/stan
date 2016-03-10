#ifndef STAN_LANG_GRAMMARS_EXPRESSION_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_EXPRESSION_GRAMMAR_DEF_HPP

#include <stan/lang/grammars/expression_grammar.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace stan {

  namespace lang {

    template <typename Iterator>
    expression_grammar<Iterator>::expression_grammar(variable_map& var_map,
                                             std::stringstream& error_msgs)
      : expression_grammar::base_type(expression_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression07_g(var_map, error_msgs, *this) {
      using boost::spirit::qi::lit;
      using boost::spirit::qi::_1;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::_val;

      // _r1 : var_origin

      expression_r.name("expression");
      expression_r
        = expression14_r(_r1)[set_expression_f(_val, _1)]
        > *(lit("||")
            > expression14_r(_r1)
              [binary_op_f(_val, _1, "||", "logical_or",
                           boost::phoenix::ref(error_msgs))]);

      expression14_r.name("expression");
      expression14_r
        = expression10_r(_r1)[set_expression_f(_val, _1)]
        > *(lit("&&")
            > expression10_r(_r1)
              [binary_op_f(_val, _1, "&&", "logical_and",
                           boost::phoenix::ref(error_msgs))]);

      expression10_r.name("expression");
      expression10_r
        = expression09_r(_r1)[set_expression_f(_val, _1)]
        > *((lit("==")
             > expression09_r(_r1)
               [binary_op_f(_val, _1, "==", "logical_eq",
                            boost::phoenix::ref(error_msgs))])
              |
              (lit("!=")
               > expression09_r(_r1)
                 [binary_op_f(_val, _1, "!=", "logical_neq",
                              boost::phoenix::ref(error_msgs))]));

      expression09_r.name("expression");
      expression09_r
        = expression07_g(_r1)[set_expression_f(_val, _1)]
        > *((lit("<=")
             > expression07_g(_r1)
               [binary_op_f(_val, _1, "<", "logical_lte",
                            boost::phoenix::ref(error_msgs))])
            | (lit("<")
               > expression07_g(_r1)
                 [binary_op_f(_val, _1, "<=", "logical_lt",
                              boost::phoenix::ref(error_msgs))])
            | (lit(">=")
               > expression07_g(_r1)
                 [binary_op_f(_val, _1, ">", "logical_gte",
                              boost::phoenix::ref(error_msgs))])
            | (lit(">")
               > expression07_g(_r1)
                 [binary_op_f(_val, _1, ">=", "logical_gt",
                              boost::phoenix::ref(error_msgs))]));
    }

  }
}
#endif
