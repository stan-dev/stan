#ifndef __STAN__VB__BASE_VB__HPP__
#define __STAN__VB__BASE_VB__HPP__

#include <ostream>

namespace stan {

  namespace vb {

    class base_vb
    {

    public:

      base_vb(std::ostream* o, std::ostream* e):
        out_stream_(o), err_stream_(e) {};

      virtual ~base_vb() {};

    protected:

      std::ostream* out_stream_;
      std::ostream* err_stream_;

      void write_error_msg_(std::ostream* error_msgs, const std::exception& e)
      {
        if (!error_msgs) return;

        *error_msgs << std::endl
                    << "[stan::vb::base_vb.hpp] encountered an error:"
                    << std::endl
                    << e.what() << std::endl << std::endl;
      };

    };

  } // vb

} // stan

#endif

