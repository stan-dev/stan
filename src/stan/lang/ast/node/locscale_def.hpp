#ifndef STAN_LANG_AST_NODE_LOCSCALE_DEF_HPP
#define STAN_LANG_AST_NODE_LOCSCALE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    locscale::locscale() { }

    locscale::locscale(const expression& loc, const expression& scale)
      : loc_(loc), scale_(scale) {  }

    bool locscale::has_loc() const {
      return !is_nil(loc_.expr_);
    }

    bool locscale::has_scale() const {
      return !is_nil(scale_.expr_);
    }

  }
}
#endif
