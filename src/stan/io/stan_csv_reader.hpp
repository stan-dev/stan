#ifndef __STAN__IO__STAN_CSV_READER_HPP__
#define __STAN__IO__STAN_CSV_READER_HPP__

#include <istream>
#include <iostream>

namespace stan {
  namespace io {

    /**
     * Reads from a Stan output csv file.
     */
    class stan_csv_reader {
    private:
      std::istream& in_;
      
    public:
      /** 
       * Default constructor.
       * 
       */
      stan_csv_reader() : in_(std::cin) {}
      
      /** 
       * Constructor taking in stream.
       *
       * Warning: does not close the input stream.
       * 
       * @param in 
       */
      stan_csv_reader(std::istream& in) : in_(in) { }
      
      /** 
       * Destructor.
       * 
       */
      ~stan_csv_reader() { }

      void parse() {
	
      }
      
      
    };
    
  }
}

#endif
