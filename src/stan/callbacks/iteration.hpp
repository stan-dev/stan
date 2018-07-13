#ifndef STAN_CALLBACKS_ITERATION_HPP
#define STAN_CALLBACKS_ITERATION_HPP

namespace stan {
  namespace callbacks {

    /**
     * <code>iteration</code> is a base class defining the interface
     * for Stan iteration callbacks.
     *
     * The callback is called from within Stan algorithms to allow
     * for the interfaces to write a message to the logger. This will
     * be called at the start of each iteration.
     */
    class iteration {
    public:
      /**
       * Called at the start of the iteration.
       *
       * Iterations are 1-indexed and this function is called
       * at the start of the iteration (before work begins).
       *
       * @param[in] iteration_number The current iteration number.
       */
      virtual void operator()(int iteration_number) {
      }

      /**
       * Virtual destructor.
       */
      virtual ~iteration() {}
    };

  }
}
#endif
