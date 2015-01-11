#ifndef STAN__GM__PARSER__STATEMENT_2_GRAMMAR__HPP
#define STAN__GM__PARSER__STATEMENT_2_GRAMMAR__HPP

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>

namespace stan { 

  namespace gm {

    template <typename Iterator>
    struct statement_grammar;

    template <typename Iterator>
    struct statement_2_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   statement(bool,var_origin,bool),
                                   whitespace_grammar<Iterator> > {
      
  
      
      statement_2_grammar(variable_map& var_map,
                          std::stringstream& error_msgs,
                          statement_grammar<Iterator>& sg);


      // global info for parses
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      
      // grammars
      expression_grammar<Iterator> expression_g;  
      statement_grammar<Iterator>& statement_g;

      // rules

      boost::spirit::qi::rule<Iterator, 
                              conditional_statement(bool,var_origin,bool),
                              whitespace_grammar<Iterator> > 
      conditional_statement_r;


      boost::spirit::qi::rule<Iterator, 
                              statement(bool,var_origin,bool), 
                              whitespace_grammar<Iterator> > 
      statement_2_r;

    };
                               

  }
}

#endif
