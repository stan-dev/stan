#include <stan/gm/grammars/var_decls_grammar_def.hpp>
#include <stan/gm/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace gm {
    template struct var_decls_grammar<pos_iterator_t>;
  }
}
