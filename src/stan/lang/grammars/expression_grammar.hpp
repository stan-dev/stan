#ifndef STAN__LANG__PARSER__EXPRESSION_GRAMMAR__HPP
#define STAN__LANG__PARSER__EXPRESSION_GRAMMAR__HPP

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/expression07_grammar.hpp>
#include <stan/lang/grammars/term_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

namespace stan { 

  namespace lang {

    template <typename Iterator>
    struct term_grammar;

    template <typename Iterator>
    struct expression_grammar;

    template <typename Iterator>
    struct expression07_grammar;

    template <typename Iterator>
    struct expression_grammar 
      : public boost::spirit::qi::grammar<Iterator,
                                          expression(var_origin),
                                          whitespace_grammar<Iterator> > {
      
      expression_grammar(variable_map& var_map,
                         std::stringstream& error_msgs);

      variable_map& var_map_;

      std::stringstream& error_msgs_;

      expression07_grammar<Iterator> expression07_g;


      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression_r;

      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression09_r;

      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression10_r;

      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression14_r;


    };

  }
}

#endif
