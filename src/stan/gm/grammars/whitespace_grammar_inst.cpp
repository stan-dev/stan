#include <stan/gm/grammars/whitespace_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct whitespace_grammar<pos_iterator_t>;
  }
}
