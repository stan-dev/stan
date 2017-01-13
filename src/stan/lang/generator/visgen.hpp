#ifndef STAN_LANG_GENERATOR_VISGEN_HPP
#define STAN_LANG_GENERATOR_VISGEN_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    /**
     * Base class for variant type visitor that generates output by
     * writing to an output stream.
     */
    struct visgen {
      /**
       * Result type for visitor pattern; always void for generators.
       */
      typedef void result_type;

      /**
       * Construct a varint type visitor for generation to the
       * specified output stream.  The specified output stream must
       * remain in scope as long as this object is created.
       *
       * @param[in] o output stream to store by reference for generation
       */
      explicit visgen(std::ostream& o) : o_(o) { }

      /**
       * Base destructor does nothing.  Specialize in subclasses.
       */
      virtual ~visgen() { }

      /**
       * Reference to output stream for generation.
       */
      std::ostream& o_;
    };



  }
}
#endif
