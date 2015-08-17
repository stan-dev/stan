#ifndef STAN_SERVICES_IO_DO_PRINT_HPP
#define STAN_SERVICES_IO_DO_PRINT_HPP

namespace stan {
  namespace services {
    namespace io {

      /**
       * Indicates whether it should print on the current iteration.
       * The function returns:
       *   true if refresh > 0 and
       *     (n == 0 or (n + 1) % refresh == 0 or special == true)
       *   false otherwise.
       * Examples:
       *   if refresh = 0, always returns false.
       *   if refresh = 10 and special == false,
       *      returns true for n = {0, 9, 19, ...},
       *      returns false for n = {1 - 8, 10 - 18, 20 - 28, ...}.
       *   if refresh = 10 and special == true, returns true for all n.
       *
       * @param n Iteration number
       * @param special When true, returns true
       * @param refresh Number of iterations to refresh
       */
      bool do_print(const int n, const bool special, const int refresh) {
        return (refresh > 0) &&
          (special || n == 0 || ((n + 1) % refresh == 0) );
      }

      /**
       * Indicates whether it should print on the current iteration.
       * The function returns:
       *   true if refresh > 0 and (n == 0 or (n + 1) % refresh == 0)
       *   false otherwise.
       * Examples:
       *   if refresh = 0, always returns false.
       *   if refresh = 10, returns true for n = {0, 9, 19, ...},
       *      returns false for n = {1 - 8, 10 - 18, 20 - 28, ...}.
       *
       * @param n Iteration number
       * @param refresh Number of iterations to refresh
       */
      bool do_print(const int n, const int refresh) {
        return do_print(n, false, refresh);
      }

    }
  }
}

#endif
