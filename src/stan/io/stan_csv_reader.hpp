#ifndef __STAN__IO__STAN_CSV_READER_HPP__
#define __STAN__IO__STAN_CSV_READER_HPP__

#include <istream>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
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
      bool nondiag_mass;
      int leapfrog_steps;
      int max_treedepth;
      double epsilon;
      double epsilon_pm;
      double delta;
      double gamma;
    };

    struct stan_csv_adaptation {
      std::string sampler;
      double step_size;
      Eigen::MatrixXd step_size_multipliers;
    };

    struct stan_csv {
      stan_csv_metadata metadata;
      Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
      stan_csv_adaptation adaptation;
      Eigen::MatrixXd samples;
    };

    /**
     * Reads from a Stan output csv file.
     */
    class stan_csv_reader {
    public:
      /** 
       * Default constructor.
       * 
       */
      stan_csv_reader() {}
      
      /** 
       * Destructor.
       * 
       */
      ~stan_csv_reader() { }

      static bool read_metadata(std::istream& in, stan_csv_metadata& metadata) {
  std::stringstream ss;
  std::string line;

  if (in.peek() != '#')
    return false;
  while (in.peek() == '#') {
    std::getline(in, line);
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
      ss >> metadata.stan_version_major;
    } else if (lhs.compare("stan_version_minor") == 0) {
      ss >> metadata.stan_version_minor;
    } else if (lhs.compare("stan_version_patch") == 0) {
      ss >> metadata.stan_version_patch;
    } else if (lhs.compare("data") == 0) {
      ss >> metadata.data;      
    } else if (lhs.compare("init") == 0) {
      std::getline(ss, metadata.init);
      boost::trim(metadata.init);
      ss.unget();
    } else if (lhs.compare("append_samples") == 0) {
      ss >> metadata.append_samples;
    } else if (lhs.compare("save_warmup") == 0) {
      ss >> metadata.save_warmup;
    } else if (lhs.compare("seed") == 0) {
      ss >> metadata.seed;
      metadata.random_seed = false;
    } else if (lhs.compare("chain_id") == 0) {
      ss >> metadata.chain_id;
    } else if (lhs.compare("iter") == 0) {
      ss >> metadata.iter;
    } else if (lhs.compare("warmup") == 0) {
      ss >> metadata.warmup;
    } else if (lhs.compare("thin") == 0) {
      ss >> metadata.thin;
    } else if (lhs.compare("equal_step_sizes") == 0) {
      ss >> metadata.equal_step_sizes;
          } else if (lhs.compare("nondiag_mass") == 0) {
      ss >> metadata.nondiag_mass;
    } else if (lhs.compare("leapfrog_steps") == 0) {
      ss >> metadata.leapfrog_steps;
    } else if (lhs.compare("max_treedepth") == 0) {
      ss >> metadata.max_treedepth;
    } else if (lhs.compare("epsilon") == 0) {
      ss >> metadata.epsilon;
    } else if (lhs.compare("epsilon_pm") == 0) {
      ss >> metadata.epsilon_pm;
    } else if (lhs.compare("delta") == 0) {
      ss >> metadata.delta;
    } else if (lhs.compare("gamma") == 0) {
      ss >> metadata.gamma;
    } else {
      std::cout << "unused option: " << lhs << std::endl;
    }
    std::getline(ss, line);
  }
  if (ss.good() == true)
    return false;
    return true;
      }
  
      static bool read_header(std::istream& in, Eigen::Matrix<std::string, Eigen::Dynamic, 1>& header) { 
  std::string line;

  if (in.peek() != 'l')
    return false;
  std::getline(in, line);
  std::stringstream ss(line);
  
  header.resize(std::count(line.begin(), line.end(), ',') + 1);
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
    header(idx++) = token;
  }
  return true;
      }

  static bool read_adaptation(std::istream& in, stan_csv_adaptation& adaptation) { 
  
    std::stringstream ss;
    std::string line;
    int lines = 0;

    if (in.peek() != '#' || in.good() == false)
      return false;
    
    while (in.peek() == '#') {
      std::getline(in, line);
      ss << line << std::endl;
      lines++;
    }
    ss.seekg(std::ios_base::beg);
       
    char comment; // Buffer for comment indicator, #
    
    // Sampler name
    ss >> comment;
    std::getline(ss, adaptation.sampler);
  
    std::replace(adaptation.sampler.begin(), 
                 adaptation.sampler.end(), 
                 '(', ' ');
    std::replace(adaptation.sampler.begin(), 
                 adaptation.sampler.end(),
                 ')', ' ');
    boost::trim(adaptation.sampler);  

    // Stepsize
    std::getline(ss, line);
  
    std::getline(ss, line, '=');
    boost::trim(line);
    ss >> adaptation.step_size;

    // Metric parameters
    std::getline(ss, line);
    std::getline(ss, line);
    std::getline(ss, line);
    
    int rows = lines - 4;
    int cols = std::count(line.begin(), line.end(), ',') + 1;
    adaptation.step_size_multipliers.resize(rows, cols);
   
    for (int row = 0; row < rows; row++) {
      
      std::stringstream line_ss;
      line_ss.str(line);
      line_ss >> comment;
      
      for (int col = 0; col < cols; col++) {
        std::string token;
        std::getline(line_ss, token, ',');
        boost::trim(token);
        adaptation.step_size_multipliers(row, col) = boost::lexical_cast<double>(token);
      }
      
      std::getline(ss, line); // Read in next line
      
    }
        
    if (ss.good())
      return false;
    else
      return true;

  }
      
      static bool read_samples(std::istream& in, Eigen::MatrixXd& samples) { 
  std::stringstream ss;
  std::string line;

  int rows = 0;  
  int cols = -1;

  if (in.peek() == '#' || in.good() == false)
    return false;

  while (in.good()) {
    bool comment_line = (in.peek() == '#');
    std::getline(in, line);
    if (!comment_line) {
      ss << line << '\n';
      int current_cols = std::count(line.begin(), line.end(), ',') + 1;
      if (cols == -1) {
        cols = current_cols;
      } else if (cols != current_cols) {
        std::cout << "Error: expected " << cols << " columns, but found " 
      << current_cols << " instead for row " << rows+1 << std::endl;
        return false;
      }
      rows++;
    }
    in.peek();
  }
  ss.seekg(std::ios_base::beg);

  if (rows > 0) {
    samples.resize(rows, cols);
    char comma;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        ss >> samples(row,col);
        if (col != cols-1)
    ss >> comma;
      }
    }
  }
  return true;
      }

      /** 
       * Parses the file.
       * 
       */
      static stan_csv parse(std::istream& in) {
  stan_csv data;
  if (!read_metadata(in, data.metadata)) {
    std::cout << "Warning: non-fatal error reading metadata" << std::endl;
  }
  if (!read_header(in, data.header)) {
    std::cout << "Error: error reading header" << std::endl;
    throw std::invalid_argument("Error with header of input file in parse");
  }
  if (!read_adaptation(in, data.adaptation)) {
    std::cout << "Warning: non-fatal error reading adapation data" << std::endl;
  }
  if (!read_samples(in, data.samples)) {
    std::cout << "Warning: non-fatal error reading samples" << std::endl;
  }
  return data;
      }
      
    };
    
  }
}

#endif
