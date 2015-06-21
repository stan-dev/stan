#ifndef STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_DEF_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
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

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/bare_type_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::lang::expr_type,
                          (stan::lang::base_expr_type, base_type_)
                          (size_t, num_dims_) )

namespace stan {

  namespace lang {

    struct set_val {
      template <class> struct result;
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
      template <typename T1, typename T2>
      void operator()(T1& lhs,
                      const T2& rhs) const {
        lhs = rhs;
      }
    };
    boost::phoenix::function<set_val> set_val_f;

    struct increment_val {
      template <class> struct result;
      template <typename F, typename T1>
      struct result<F(T1)> { typedef void type; };
      template <typename T1>
      void operator()(T1& lhs) const {
        lhs += 1;
      }
    };
    boost::phoenix::function<increment_val> increment_val_f;

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
      bare_type_r
        %= type_identifier_r
        >> array_dims_r;

      type_identifier_r.name("bare type identifier\n"
                "  legal values: void, int, real, vector, row_vector, matrix");

      type_identifier_r
        %= lit("void")[set_val_f(_val, VOID_T)]
        | lit("int")[set_val_f(_val, INT_T)]
        | lit("real")[set_val_f(_val, DOUBLE_T)]
        | lit("vector")[set_val_f(_val, VECTOR_T)]
        | lit("row_vector")[set_val_f(_val, ROW_VECTOR_T)]
        | lit("matrix")[set_val_f(_val, MATRIX_T)];

      array_dims_r.name("array dimensions,\n"
             "    e.g., empty (not an array) [] (1D array) or [,] (2D array)");
      array_dims_r
        %= eps[_val = 0]
        >> - (lit('[')[set_val_f(_val, 1)]
              > *(lit(',')[increment_val_f(_val)])
              > end_bare_types_r);

      end_bare_types_r.name("comma to indicate more dimensions"
                            " or ] to end type declaration");
      end_bare_types_r
        %= lit(']');
    }

  }
}
#endif
