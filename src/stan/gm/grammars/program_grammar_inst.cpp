#include <stan/gm/grammar/program_grammar_def.hpp>
#include <stan/gm/grammar/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct program_grammar<pos_iterator_t>;
  }
}
