#include <stan/gm/parser/expression_grammar_def.hpp>
#include <stan/gm/parser/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct expression_grammar<pos_iterator_t>;
  }
}
