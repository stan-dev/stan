#ifndef STAN__LANG__PARSER__EXPRESSION_GRAMMAR07__HPP__
#define STAN__LANG__PARSER__EXPRESSION_GRAMMAR07__HPP__

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/term_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

namespace stan { 

  namespace lang {

    template <typename Iterator>
    struct term_grammar;

    template <typename Iterator>
    struct expression_grammar;

    template <typename Iterator>
    struct expression07_grammar 
      : public boost::spirit::qi::grammar<Iterator,
                                          expression(var_origin),
                                          whitespace_grammar<Iterator> > {
      
      expression07_grammar(variable_map& var_map,
                           std::stringstream& error_msgs,
                           Iterator& it,
                           expression_grammar<Iterator>& eg);

      // global parser information
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      Iterator& it_;

      // nested grammars
      term_grammar<Iterator> term_g;
      
      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression07_r;

    };

  }
}

#endif
