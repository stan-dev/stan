#ifndef STAN_INTERFACE_CALLBACKS_WRITER_MESSAGES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_MESSAGES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      class messages: public base_writer {
      private:
        std::ostream *o_;
        const bool has_stream_;
        const std::string prefix_;

      public:
        messages(std::ostream *o, std::string prefix)
          : o_(o), has_stream_(o != 0), prefix_(prefix) { }

        explicit messages(std::ostream *o)
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

        bool is_writing() const {
          return has_stream_;
        }
      };
    }
  }
}

#endif
