#ifndef STAN__VB__BASE_VB__HPP
#define STAN__VB__BASE_VB__HPP

#include <ostream>
#include <string>

namespace stan {

  namespace vb {

    class base_vb {

    public:

      base_vb(std::ostream* o, std::ostream* e, std::string name):
        out_stream_(o), err_stream_(e), name_(name) {};

      virtual ~base_vb() {};

    protected:

      std::ostream* out_stream_;
      std::ostream* err_stream_;
      
      std::string name_;

      void write_error_msg_(std::ostream* error_msgs, const std::exception& e) {
        if (!error_msgs) return;

        *error_msgs << std::endl
                    << "[stan::vb::" << name_ << "] encountered an error:"
                    << std::endl
                    << e.what() << std::endl << std::endl;
      };

    };

  } // vb

} // stan

#endif

