#ifndef STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/expression07_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace stan {
  namespace lang {

    template <typename Iterator>
    struct block_var_decls_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<scope>,
                                   std::vector<block_var_decl>,
                                   whitespace_grammar<Iterator> > {
      block_var_decls_grammar(variable_map& var_map,
                              std::stringstream& error_msgs,
                              const io::program_reader& reader);

      const io::program_reader& reader_;
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      expression_grammar<Iterator> expression_g;
      expression07_grammar<Iterator> expression07_g;  // disallows comparisons

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<scope>,
                              std::vector<block_var_decl>,
                              whitespace_grammar<Iterator> >
      var_decls_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      array_var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      single_var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              block_var_type(scope),
                              whitespace_grammar<Iterator> >
      element_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              double_block_type(scope),
                              whitespace_grammar<Iterator> >
      double_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              int_block_type(scope),
                              whitespace_grammar<Iterator> >
      int_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              row_vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      row_vector_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      vector_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              cholesky_factor_corr_block_type(scope),
                              whitespace_grammar<Iterator> >
      cholesky_factor_corr_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              cholesky_factor_cov_block_type(scope),
                              whitespace_grammar<Iterator> >
      cholesky_factor_cov_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              corr_matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      corr_matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              cov_matrix_block_type(scope),
                              whitespace_grammar<Iterator> >
      cov_matrix_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              ordered_block_type(scope),
                              whitespace_grammar<Iterator> >
      ordered_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              positive_ordered_block_type(scope),
                              whitespace_grammar<Iterator> >
      positive_ordered_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              simplex_block_type(scope),
                              whitespace_grammar<Iterator> >
      simplex_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              unit_vector_block_type(scope),
                              whitespace_grammar<Iterator> >
      unit_vector_type_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_name_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      opt_def_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      def_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_double_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_int_r;

      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      dim1_r;

      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      int_data_expr_r;

      boost::spirit::qi::rule<Iterator,
                              std::vector<expression>(scope),
                              whitespace_grammar<Iterator> >
      dims_r;
    };

  }
}
#endif
