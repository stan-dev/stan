#include <stan/lang/grammars/var_decls_grammar_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>

namespace stan {
  namespace lang {
    template struct var_decls_grammar<pos_iterator_t>;
  }
}
