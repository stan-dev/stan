#ifndef STAN_CALLBACKS_WRITER_HPP
#define STAN_CALLBACKS_WRITER_HPP

#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace callbacks {

    /**
     * <code>writer</code> is an abstract base class defining the interface
     * for Stan writer callbacks.
     *
     * The Stan API writes structured output to implementations of
     * this class.
     */
    class writer {
    public:
      /**
       * Writes a set of names.
       *
       * @param[in] names Names in a std::vector
       */
      virtual void operator()(const std::vector<std::string>& names) = 0;

      /**
       * Writes a set of values.
       *
       * @param[in] state Values in a std::vector
       */
      virtual void operator()(const std::vector<double>& state) = 0;

      /**
       * Writes blank input.
       */
      virtual void operator()() = 0;

      /**
       * Writes a string.
       *
       * @param[in] message A string
       */
      virtual void operator()(const std::string& message) = 0;

      /**
       * Destructor.
       *
       * Virtual destructor to avoid compiler warnings
       *
       */
      virtual ~writer() {}
    };

  }
}
#endif
