#ifndef __STAN__IO__STAN_CSV_READER_HPP__
#define __STAN__IO__STAN_CSV_READER_HPP__

#include <istream>
#include <iostream>

namespace stan {
  namespace io {

    // FIXME: should consolidate with the options from the command line in stan::gm
    struct stan_csv_metadata {
      int stan_version_major;
      int stan_version_minor;
      int stan_version_patch;
      
      std::string data;
      std::string init;
      bool append_samples;
      bool save_warmup;
      size_t seed;
      bool random_seed;
      size_t chain_id;
      size_t iter;
      size_t warmup;
      size_t thin;
      bool equal_step_sizes;
      int leapfrog_steps;
      int max_treedepth;
      double epsilon;
      double epsilon_pm;
      double delta;
      double gamma;
    };

    /**
     * Reads from a Stan output csv file.
     */
    class stan_csv_reader {
    private:
      std::istream& in_;
      stan_csv_metadata metadata_;

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

      void read_metadata() {
	
      }
      void read_header() { }
      void read_adaptation() { }
      void read_samples() { }

      /** 
       * Parses the file.
       * 
       */
      void parse() {
	// read_metadata()
	// read_header()
	// read_adaptation()
	// read_samples()
      }
      
      stan_csv_metadata metadata() {
	return metadata_;
      }
      
    };
    
  }
}

#endif
