#ifndef STAN__GM__PARSER__TERM_GRAMMAR__HPP
#define STAN__GM__PARSER__TERM_GRAMMAR__HPP

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
    struct term_grammar;

    template <typename Iterator>
    struct expression_grammar;

    template <typename Iterator>
    struct term_grammar 
      : public boost::spirit::qi::grammar<Iterator,
                                          expression(var_origin),
                                          whitespace_grammar<Iterator> > {

      term_grammar(variable_map& var_map,
                   std::stringstream& error_msgs,
                   expression_grammar<Iterator>& eg);

      variable_map& var_map_;

      std::stringstream& error_msgs_;

      stan::gm::expression_grammar<Iterator>& expression_g;


      boost::spirit::qi::rule<Iterator, 
                              std::vector<expression>(var_origin), 
                              whitespace_grammar<Iterator> > 
      args_r;


      boost::spirit::qi::rule<Iterator, 
                              std::vector<expression>(var_origin), 
                              whitespace_grammar<Iterator> > 
      dims_r;


      boost::spirit::qi::rule<Iterator, 
                              double_literal(),
                              whitespace_grammar<Iterator> > 
      double_literal_r;


      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<bool>, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      factor_r;


      boost::spirit::qi::rule<Iterator, 
                              fun(var_origin), 
                              whitespace_grammar<Iterator> > 
      fun_r;

      boost::spirit::qi::rule<Iterator, 
                              integrate_ode(var_origin), 
                              whitespace_grammar<Iterator> > 
      integrate_ode_r;


      boost::spirit::qi::rule<Iterator, 
                              std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;


      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      indexed_factor_r;


      boost::spirit::qi::rule<Iterator,
                              int_literal(), 
                              whitespace_grammar<Iterator> > 
      int_literal_r;


      boost::spirit::qi::rule<Iterator,
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      negated_factor_r;


      boost::spirit::qi::rule<Iterator,
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      exponentiated_factor_r;


      boost::spirit::qi::rule<Iterator, 
                              expression(var_origin), 
                              whitespace_grammar<Iterator> > 
      term_r;


      boost::spirit::qi::rule<Iterator, 
                              variable(),
                              whitespace_grammar<Iterator> > 
      variable_r;

    };

  }
}

#endif
