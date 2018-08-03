#include <test/unit/lang/parser/test_bare_type_grammar_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace lang {
    template struct test_bare_type_grammar<pos_iterator_t>;
  }
}
