#ifndef STAN_INTERFACE_CALLBACKS_WRITER_CSV_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_CSV_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math/prim/scal/meta/index_type.hpp>
#include <stan/math/prim/arr/meta/index_type.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * Writes out a vector as string.
       */
      class csv: public base_writer {
      private:
        std::ostream *o_;
        const bool has_stream_;
        const std::string prefix_;

      public:
        /**
         * Construct an object.
         *
         * @param o pointer to stream. Will accept 0.
         * @param prefix Prefix string
         */
        csv(std::ostream *o, std::string prefix)
          : o_(o), has_stream_(o != 0), prefix_(prefix) { }

        /**
         * Print vector as csv.
         *
         * Uses the insertion operator to write out the elements
         * as comma separated values, flushing the buffer after the
         * line is complete
         *
         * @tparam T type of element
         * @param x vector of type T
         */
        template <class T>
        void operator()(const std::vector<T>& x) {
          typedef typename stan::math::index_type<std::vector<T> >::type idx_t;
          if (!has_stream_)
            return;

          if (x.size() != 0) {
            *o_ << x[0];
            for (idx_t n = 1; n < x.size(); n++) {
              *o_ << "," << x[n];
            }
          }
          *o_ << std::endl;
        }

        /**
         * Print single string with a prefix
         *
         * Uses the insertion operator to write out a string
         * as comma separated values, flushing the buffer after the
         * line is complete
         *
         * @param message string to print with prefix in front
         */
        void operator()(const std::string& message) {
          if (!has_stream_)
            return;
          *o_ << prefix_ << message << std::endl;
        }

        /**
         * Prints a blank line. No prefix, no nothing.
         *
         */
        void operator()() {
          if (!has_stream_)
            return;
          *o_ << std::endl;
        }
      };

    }
  }
}

#endif
