#ifndef STAN_LANG_GRAMMARS_STATEMENT_2_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_STATEMENT_2_GRAMMAR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/common_adaptors_def.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <stan/lang/grammars/statement_grammar.hpp>
#include <stan/lang/grammars/statement_2_grammar.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <sstream>

namespace stan {

  namespace lang {

    template <typename Iterator>
    statement_2_grammar<Iterator>::statement_2_grammar(variable_map& var_map,
                                           std::stringstream& error_msgs,
                                           statement_grammar<Iterator>& sg)
      : statement_2_grammar::base_type(statement_2_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map, error_msgs),
        statement_g(sg) {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;
      using boost::spirit::qi::labels::_r3;
      using boost::spirit::qi::labels::_r4;

      // _r1 true if sample_r allowed (inherited)
      // _r2 source of variables allowed for assignments
      // set to true if sample_r are allowed
      statement_2_r.name("statement");
      statement_2_r %= conditional_statement_r(_r1, _r2, _r3, _r4);

      conditional_statement_r.name("if-else statement");
      conditional_statement_r
        = (lit("if")  >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('(')
        > expression_g(_r2)
          [add_conditional_condition_f(_val, _1, _pass,
                                       boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > statement_g(_r1, _r2, _r3, _r4)
          [add_conditional_body_f(_val, _1)]
        > * (((lit("else") >> no_skip[!char_("a-zA-Z0-9_")])
              >> (lit("if")  >> no_skip[!char_("a-zA-Z0-9_")]))
             > lit('(')
             > expression_g(_r2)
               [add_conditional_condition_f(_val, _1, _pass,
                                            boost::phoenix::ref(error_msgs_))]
             > lit(')')
             > statement_g(_r1, _r2, _r3, _r4)
               [add_conditional_body_f(_val, _1)])
        > -((lit("else") >> no_skip[!char_("a-zA-Z0-9_")])
            > statement_g(_r1, _r2, _r3, _r4)
              [add_conditional_body_f(_val, _1)]);
    }

  }
}
#endif
