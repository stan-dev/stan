#ifndef STAN__COMMON__RECORDER__NO_OP_HPP
#define STAN__COMMON__RECORDER__NO_OP_HPP

#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace common {
    namespace recorder {
      
      /**
       * Does nothing
       */
      class no_op {
      public:
        /**
         * Do nothing with vector
         *
         * @tparam T type of element
         * @param x vector of type T
         */
        template <class T>
        void operator()(const std::vector<T>& x) { }
      
        /**
         * Do nothing with a string.
         * 
         * @param x string to print with prefix in front
         */
        void operator()(const std::string x) { }
      
        /**
         * Do nothing
         *
         */
        void operator()() { }
      
        /**
         * Indicator function for whether the instance is recording.
         */
        bool is_recording() const {
          return false;
        }
      };


    }
  }
}

#endif
