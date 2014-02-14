#ifndef __STAN__UI__WRITE_ERROR_MSG_HPP__
#define __STAN__UI__WRITE_ERROR_MSG_HPP__

#include <ostream>
#include <stdexcept>

namespace stan {

  namespace ui {
       
    void write_error_msg(std::ostream* error_stream,
                         const std::exception& e) {
      
      if (!error_stream) return;
      
      *error_stream << std::endl
                    << "Informational Message: The current Metropolis proposal is about to be"
                    << " rejected becuase of the following issue:"
                    << std::endl
                    << e.what() << std::endl
                    << "If this warning occurs sporadically, such as for highly constrained"
                    << " variable types like covariance matrices, then the sampler is fine,"
                    << std::endl
                    << "but if this warning occurs often then your model may be either"
                    << " severely ill-conditioned or misspecified."
                    << std::endl;
    }
    

  } // namespace ui

} // namespace stan

#endif
