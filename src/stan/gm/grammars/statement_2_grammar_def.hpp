#ifndef STAN__GM__PARSER__STATEMENT_2_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__STATEMENT_2_GRAMMAR_DEF__HPP

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
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/statement_2_grammar.hpp>
#include <stan/gm/grammars/common_adaptors_def.hpp>


namespace stan {

  namespace gm {

    struct add_conditional_condition {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(conditional_statement& cs,
                      const expression& e,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!e.expression_type().is_primitive()) {
          error_msgs << "conditions in if-else statement must be primitive int or real;"
                     << " found type=" << e.expression_type() << std::endl;
          pass = false;
          return;
        }
        cs.conditions_.push_back(e);
        pass = true;
        return;
      }               
    };
    boost::phoenix::function<add_conditional_condition> add_conditional_condition_f;

    struct add_conditional_body {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(conditional_statement& cs,
                      const statement& s) const {
        cs.bodies_.push_back(s);
      }
    };
    boost::phoenix::function<add_conditional_body> add_conditional_body_f;

 

    template <typename Iterator>
    statement_2_grammar<Iterator>::statement_2_grammar(variable_map& var_map,
                                                       std::stringstream& error_msgs,
                                                       statement_grammar<Iterator>& sg)
      : statement_2_grammar::base_type(statement_2_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map,error_msgs),
        statement_g(sg)
    {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;

      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;
      using boost::spirit::qi::labels::_r3;

      // _r1 true if sample_r allowed (inherited)
      // _r2 source of variables allowed for assignments
      // set to true if sample_r are allowed
      statement_2_r.name("statement");
      statement_2_r
        %= conditional_statement_r(_r1,_r2,_r3)
        ;

      
      conditional_statement_r.name("if-else statement");
      conditional_statement_r
        = lit("if")
        > lit('(')
        > expression_g(_r2)
          [add_conditional_condition_f(_val,_1,_pass,
                                       boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > statement_g(_r1,_r2,_r3)
          [add_conditional_body_f(_val,_1)]
        > * (( lit("else")
               >> lit("if") )
             > lit('(')
             > expression_g(_r2)
               [add_conditional_condition_f(_val,_1,_pass,
                                            boost::phoenix::ref(error_msgs_))]
             > lit(')')
             > statement_g(_r1,_r2,_r3)
               [add_conditional_body_f(_val,_1)]
             )
        > - (lit("else") 
             > statement_g(_r1,_r2,_r3)
               [add_conditional_body_f(_val,_1)]
             )
        ;

    }

  }
}
#endif
