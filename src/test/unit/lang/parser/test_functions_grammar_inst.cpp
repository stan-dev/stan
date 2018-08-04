#include <test/unit/lang/parser/test_functions_grammar_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace lang {
    template struct test_functions_grammar<pos_iterator_t>;
  }
}
