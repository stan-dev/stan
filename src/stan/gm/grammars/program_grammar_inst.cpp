#include <stan/gm/grammars/program_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct program_grammar<pos_iterator_t>;
  }
}
