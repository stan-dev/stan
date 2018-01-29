#ifndef STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/expression07_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <vector>

namespace stan {
  namespace lang {

    template <typename Iterator>
    struct block_var_decls_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<scope>,
                                   std::vector<block_var_decl>,
                                   whitespace_grammar<Iterator> > {
      block_var_decls_grammar(variable_map& var_map,
                              std::stringstream& error_msgs);

      variable_map& var_map_;
      std::stringstream& error_msgs_;
      expression_grammar<Iterator> expression_g;
      expression07_grammar<Iterator> expression07_g;  // disallows comparisons

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<scope>,
                              std::vector<block_var_decl>,
                              whitespace_grammar<Iterator> >
      block_var_decls_r;

      boost::spirit::qi::rule<Iterator,
                               std::vector<expression>(scope),
                               whitespace_grammar<Iterator> >
      array_dims_r;

      // boost::spirit::qi::rule<Iterator,
      //                         boost::spirit::qi::locals<bool>,
      //                         block_array_var_decl(scope),
      //                         whitespace_grammar<Iterator> >
      // block_array_var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_type(scope),
                              whitespace_grammar<Iterator> >
      block_el_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      block_el_var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      block_el_var_decl_sub_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      block_var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              cholesky_factor_block_type(scope),
                              whitespace_grammar<Iterator> >
      cholesky_factor_type_r;

      boost::spirit::qi::rule<Iterator,
                              cholesky_corr_block_type(scope),
                              whitespace_grammar<Iterator> >
      cholesky_corr_type_r;

      boost::spirit::qi::rule<Iterator,
                              corr_matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      corr_matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              cov_matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      cov_matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              double_block_type(scope),
                              whitespace_grammar<Iterator> >
      double_type_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_name_r;
      
      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      int_data_expr_r;

      boost::spirit::qi::rule<Iterator,
                              int_block_type(scope),
                              whitespace_grammar<Iterator> >
      int_type_r;

      boost::spirit::qi::rule<Iterator,
                              matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      opt_def_r;

      boost::spirit::qi::rule<Iterator,
                              ordered_block_type(scope),
                              whitespace_grammar<Iterator> >
      ordered_type_r;

      boost::spirit::qi::rule<Iterator,
                              positive_ordered_block_type(scope),
                              whitespace_grammar<Iterator> >
      positive_ordered_type_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_double_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_int_r;

      boost::spirit::qi::rule<Iterator,
                              row_vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      row_vector_type_r;

      boost::spirit::qi::rule<Iterator,
                              simplex_block_type(scope),
                              whitespace_grammar<Iterator> >
      simplex_type_r;

      boost::spirit::qi::rule<Iterator,
                              unit_vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      unit_vector_type_r;

      boost::spirit::qi::rule<Iterator,
                              vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      vector_type_r;
    };

  }
}
#endif
