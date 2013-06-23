#ifndef __STAN__GM__PARSER__EXPRESSION_GRAMMAR__HPP__
#define __STAN__GM__PARSER__EXPRESSION_GRAMMAR__HPP__

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/term_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>

namespace stan { 

  namespace gm {

    template <typename Iterator>
    struct term_grammar;

    template <typename Iterator>
    struct expression_grammar;

    template <typename Iterator>
    struct expression_grammar 
      : public boost::spirit::qi::grammar<Iterator,
                                          expression(var_origin),
                                          whitespace_grammar<Iterator> > {

      expression_grammar(variable_map& var_map,
                         std::stringstream& error_msgs,
                         bool allow_lte = true);

      variable_map& var_map_;

      std::stringstream& error_msgs_;

      term_grammar<Iterator> term_g;



      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression_r;

      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      expression07_r;

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
