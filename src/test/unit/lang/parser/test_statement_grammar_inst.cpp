#include <test/unit/lang/parser/test_statement_grammar_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace lang {
    template struct test_statement_grammar<pos_iterator_t>;
  }
}
