#ifndef STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_DEF_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/bare_type_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::lang::expr_type,
                          (stan::lang::base_expr_type, base_type_)
                          (size_t, num_dims_) )

namespace stan {

  namespace lang {

    template <typename Iterator>
    bare_type_grammar<Iterator>::bare_type_grammar(variable_map& var_map,
                                           std::stringstream& error_msgs)
      : bare_type_grammar::base_type(bare_type_r),
        var_map_(var_map),
        error_msgs_(error_msgs) {
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::_val;

      bare_type_r.name("bare type definition\n"
               "   (no dimensions or constraints, just commas,\n"
               "   e.g. real[,] for a 2D array or int for a single integer,\n"
               "   and constrained types such as cov_matrix not allowed)");
      bare_type_r %= type_identifier_r >> array_dims_r;

      type_identifier_r.name("bare type identifier\n"
                "  legal values: void, int, real, vector, row_vector, matrix");
      type_identifier_r
        %= lit("void")[assign_lhs_f(_val, VOID_T)]
        | lit("int")[assign_lhs_f(_val, INT_T)]
        | lit("real")[assign_lhs_f(_val, DOUBLE_T)]
        | lit("vector")[assign_lhs_f(_val, VECTOR_T)]
        | lit("row_vector")[assign_lhs_f(_val, ROW_VECTOR_T)]
        | lit("matrix")[assign_lhs_f(_val, MATRIX_T)];

      array_dims_r.name("array dimensions,\n"
             "    e.g., empty (not an array) [] (1D array) or [,] (2D array)");
      array_dims_r
        %= eps[assign_lhs_f(_val, static_cast<size_t>(0))]
        >> -(lit('[')[assign_lhs_f(_val, static_cast<size_t>(1))]
             > *(lit(',')[increment_size_t_f(_val)])
             > end_bare_types_r);

      end_bare_types_r.name("comma to indicate more dimensions"
                            " or ] to end type declaration");
      end_bare_types_r %= lit(']');
    }

  }
}
#endif
