#include <stan/gm/grammars/expression_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct expression_grammar<lp_iterator>;
  }
}
