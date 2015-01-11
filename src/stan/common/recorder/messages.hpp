#ifndef STAN__COMMON__RECORDER__MESSAGES_HPP
#define STAN__COMMON__RECORDER__MESSAGES_HPP

#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace common {
    namespace recorder {
      class messages {
      private: 
        std::ostream *o_;
        const bool has_stream_;
        const std::string prefix_;
        
      public:
        messages(std::ostream *o, std::string prefix) 
          : o_(o), has_stream_(o != 0), prefix_(prefix) { }
        
        messages(std::ostream *o) 
          : o_(o), has_stream_(o != 0), prefix_("") { }
        
        template <class T>
        void operator()(const std::vector<T>& x) {
          // no op
        }

        void operator()(const std::string x) {
          if (!has_stream_)
            return;
          *o_ << prefix_ << x << std::endl;
        }

        void operator()() {
          if (!has_stream_)
            return;
          *o_ << std::endl;
        }

        bool is_recording() const {
          return has_stream_;
        }

      };
    }
  }
}

#endif
