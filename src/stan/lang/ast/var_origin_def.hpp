#ifndef STAN_LANG_AST_VAR_ORIGIN_DEF_HPP
#define STAN_LANG_AST_VAR_ORIGIN_DEF_HPP

#include <stan/lang/ast/origin_block.hpp>
#include <stan/lang/ast/var_origin.hpp>

namespace stan {
  namespace lang {

    var_origin::var_origin()
      : program_block_(model_name_origin), is_local_(false) { }

    var_origin::var_origin(const origin_block program_block)
      : program_block_(program_block), is_local_(false) { }

    var_origin::var_origin(const origin_block program_block,
                           const bool is_local)
      : program_block_(program_block), is_local_(is_local) { }

  }
}
#endif
