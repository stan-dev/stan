#include <stan/gm/parser/statement_grammar_def.hpp>
#include <stan/gm/parser/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct statement_grammar<pos_iterator_t>;
  }
}
