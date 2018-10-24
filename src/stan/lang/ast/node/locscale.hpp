#ifndef STAN_LANG_AST_NODE_LOCSCALE_HPP
#define STAN_LANG_AST_NODE_LOCSCALE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * AST structure for a locscale object with a loc and scale value.
     */
    struct locscale {
      /**
       * Location of loc-scale pair with <code>nil</code> value if only
       * scale.
       */
      expression loc_;

      /**
       * Scale of loc-scale pair with <code>nil</code> value if only
       * location.
       */
      expression scale_;

      /**
       * Construct a default locscale object.
       */
      locscale();

      /**
       * Construct a locscale object with the specified location and scale.
       *
       * @param loc location
       * @param scale scale
       */
      locscale(const expression& loc, const expression& scale);

      /**
       * Return true if the location is non-nil.
       *
       * @return true if there is a location
       */
      bool has_loc() const;

      /**
       * Return true if the scale is non-nil.
       *
       * @return true if there is a scale
       */
      bool has_scale() const;
    };

  }
}
#endif
