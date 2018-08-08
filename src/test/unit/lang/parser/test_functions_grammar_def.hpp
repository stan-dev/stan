#ifndef STAN_LANG_GRAMMARS_TEST_FUNCTIONS_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_TEST_FUNCTIONS_GRAMMAR_DEF_HPP

#include <test/unit/lang/parser/test_functions_grammar.hpp>

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
#include <vector>

namespace stan {

  namespace lang {

    template <typename Iterator>
    test_functions_grammar<Iterator>::test_functions_grammar(
                                            const io::program_reader& reader,
                                            variable_map& var_map,
                                            std::stringstream& error_msgs)
      : test_functions_grammar::base_type(test_functions_r),
        reader_(reader),
        var_map_(var_map),
        functions_declared_(),
        functions_defined_(),
        error_msgs_(error_msgs),
        statement_g(var_map_, error_msgs_),
        bare_type_g(error_msgs_),
        functions_g(var_map_, error_msgs_, false) {

      test_functions_r.name("test functions");
      test_functions_r
        %= functions_g;
    }

  }
}
#endif
