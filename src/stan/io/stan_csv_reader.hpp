#ifndef __STAN__IO__STAN_CSV_READER_HPP__
#define __STAN__IO__STAN_CSV_READER_HPP__

#include <istream>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <stan/math/matrix.hpp>

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
      Eigen::Matrix<std::string, Eigen::Dynamic, 1> header_;

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

      bool read_metadata() {
	std::stringstream ss;
	std::string line;

	if (in_.peek() != '#')
	  return false;
	while (in_.peek() == '#') {
	  std::getline(in_, line);
	  ss << line << '\n';
	}
	ss.seekg(std::ios_base::beg);	

	char comment;
	std::string lhs;

	// skip first two lines
	std::getline(ss, line);
	std::getline(ss, line);

	while (ss.good()) {
	  ss >> comment;
	  std::getline(ss, lhs, '=');
	  boost::trim(lhs);
	  if (lhs.compare("") == 0) { // no-op
	  } else if (lhs.compare("stan_version_major") == 0) {
	    ss >> metadata_.stan_version_major;
	  } else if (lhs.compare("stan_version_minor") == 0) {
	    ss >> metadata_.stan_version_minor;
	  } else if (lhs.compare("stan_version_patch") == 0) {
	    ss >> metadata_.stan_version_patch;
	  } else if (lhs.compare("data") == 0) {
	    ss >> metadata_.data;
	  } else if (lhs.compare("init") == 0) {
	    ss >> metadata_.init;
	  } else if (lhs.compare("append_samples") == 0) {
	    ss >> metadata_.append_samples;
	  } else if (lhs.compare("save_warmup") == 0) {
	    ss >> metadata_.save_warmup;
	  } else if (lhs.compare("seed") == 0) {
	    ss >> metadata_.seed;
	  } else if (lhs.compare("chain_id") == 0) {
	    ss >> metadata_.chain_id;
	  } else if (lhs.compare("iter") == 0) {
	    ss >> metadata_.iter;
	  } else if (lhs.compare("warmup") == 0) {
	    ss >> metadata_.warmup;
	  } else if (lhs.compare("thin") == 0) {
	    ss >> metadata_.thin;
	  } else if (lhs.compare("equal_step_sizes") == 0) {
	    ss >> metadata_.equal_step_sizes;
	  } else if (lhs.compare("leapfrog_steps") == 0) {
	    ss >> metadata_.leapfrog_steps;
	  } else if (lhs.compare("max_treedepth") == 0) {
	    ss >> metadata_.max_treedepth;
	  } else if (lhs.compare("epsilon") == 0) {
	    ss >> metadata_.epsilon;
	  } else if (lhs.compare("epsilon_pm") == 0) {
	    ss >> metadata_.epsilon_pm;
	  } else if (lhs.compare("delta") == 0) {
	    ss >> metadata_.delta;
	  } else if (lhs.compare("gamma") == 0) {
	    ss >> metadata_.gamma;
	  } else {
	    std::cout << "unused option: " << lhs << std::endl;
	  }
	  std::getline(ss, line);
	}
	if (ss.good() == false)
	  return false;
  	return true;
      }
  
      bool read_header() { 
	std::string line;

	if (in_.peek() != 'l')
	  return false;
	std::getline(in_, line);
	std::stringstream ss(line);
	
	header_.resize(std::count(line.begin(), line.end(), ',') + 1);
	int idx = 0;
	while (ss.good()) {
	  std::string token;
	  std::getline(ss, token, ',');
	  boost::trim(token);
	  
	  int pos = token.find('.');
	  if (pos > 0) {
	    token.replace(pos, 1, "[");
	    std::replace(token.begin(), token.end(), '.', ',');
	    token += "]";
	  }
	  header_(idx++) = token;
	}
	return false;
      }

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

      Eigen::Matrix<std::string, Eigen::Dynamic, 1> header() {
	return header_;
      }
      
    };
    
  }
}

#endif
