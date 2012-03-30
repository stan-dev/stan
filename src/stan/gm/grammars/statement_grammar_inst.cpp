#include <stan/gm/grammars/statement_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct statement_grammar<pos_iterator_t>;
  }
}
