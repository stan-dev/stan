#ifndef STAN_LANG_GRAMMARS_TEST_EXPRESSION_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_TEST_EXPRESSION_GRAMMAR_DEF_HPP

#include <test/unit/lang/parser/test_expression_grammar.hpp>

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/qi.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace stan {

  namespace lang {

    template <typename Iterator>
    test_expression_grammar<Iterator>::test_expression_grammar(
                                            const io::program_reader& reader,
                                            variable_map& var_map,
                                            std::stringstream& error_msgs)
      : test_expression_grammar::base_type(test_expression_r),
        reader_(reader),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map_,error_msgs_) {
      using boost::spirit::qi::eps;
      using boost::spirit::qi::labels::_a;

      test_expression_r.name("test expression");
      test_expression_r
        %= eps[set_var_scope_f(_a, derived_origin)]
        > expression_g(_a);

    }

  }
}
#endif
