#ifndef __STAN__GM__PARSER__STATEMENT_GRAMMAR__HPP__
#define __STAN__GM__PARSER__STATEMENT_GRAMMAR__HPP__

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/parser/whitespace_grammar.hpp>
#include <stan/gm/parser/expression_grammar.hpp>
#include <stan/gm/parser/var_decls_grammar.hpp>

namespace stan { 

  namespace gm {

    template <typename Iterator>
    struct statement_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   statement(bool,var_origin),
                                   whitespace_grammar<Iterator> > {

  
      
      statement_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);


      // global info for parses
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      
      // grammars
      expression_grammar<Iterator> expression_g;  
      var_decls_grammar<Iterator> var_decls_g;

      // rules
      boost::spirit::qi::rule<Iterator, 
                              assignment(), 
                              whitespace_grammar<Iterator> > 
      assignment_r;

      boost::spirit::qi::rule<Iterator, 
                              std::vector<expression>(), 
                              whitespace_grammar<Iterator> > 
      dims_r;

      boost::spirit::qi::rule<Iterator, 
                              distribution(),
                              whitespace_grammar<Iterator> >
      distribution_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<std::string>, 
                              for_statement(bool,var_origin), 
                              whitespace_grammar<Iterator> > 
      for_statement_r;

      boost::spirit::qi::rule<Iterator, 
                              std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;

      boost::spirit::qi::rule<Iterator, 
                              std::vector<var_decl>(), 
                              whitespace_grammar<Iterator> >
      local_var_decls_r;

      boost::spirit::qi::rule<Iterator, 
                              no_op_statement(), 
                              whitespace_grammar<Iterator> > 
      no_op_statement_r;

      boost::spirit::qi::rule<Iterator, 
                              std::vector<expression>(),
                              whitespace_grammar<Iterator> > 
      opt_dims_r;

      boost::spirit::qi::rule<Iterator,
                              range(), 
                              whitespace_grammar<Iterator> > 
      range_r;

      boost::spirit::qi::rule<Iterator, 
                              sample(bool),
                              whitespace_grammar<Iterator> > 
      sample_r;

      boost::spirit::qi::rule<Iterator, 
                              statement(bool,var_origin), 
                              whitespace_grammar<Iterator> > 
      statement_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<std::vector<var_decl> >,
                              statements(bool,var_origin), 
                              whitespace_grammar<Iterator> >
      statement_seq_r;

      boost::spirit::qi::rule<Iterator, 
                              range(), 
                              whitespace_grammar<Iterator> > 
      truncation_range_r;

      boost::spirit::qi::rule<Iterator, 
                              variable_dims(),
                              whitespace_grammar<Iterator> > 
      var_lhs_r;

    };
                               

  }
}

#endif
