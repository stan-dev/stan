#ifndef STAN_LANG_GRAMMARS_TEST_BARE_TYPE_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_TEST_BARE_TYPE_GRAMMAR_DEF_HPP

#include <test/unit/lang/parser/test_bare_type_grammar.hpp>

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
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
    test_bare_type_grammar<Iterator>::test_bare_type_grammar(
                                            const io::program_reader& reader,
                                            std::stringstream& error_msgs)
      : test_bare_type_grammar::base_type(test_bare_type_r),
        reader_(reader),
        error_msgs_(error_msgs),
        bare_type_g(error_msgs_) {

      test_bare_type_r.name("test bare_type");
      test_bare_type_r
        %= bare_type_g;
    }

  }
}
#endif
