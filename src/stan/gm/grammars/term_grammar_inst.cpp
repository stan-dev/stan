#include <stan/gm/grammars/term_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct term_grammar<pos_iterator_t>;
  }
}
