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
      
      std::string model;
      std::string data;
      std::string init;
      size_t chain_id;
      size_t seed;
      bool random_seed;
      size_t num_samples;
      size_t num_warmup;
      bool save_warmup;
      size_t thin;
      bool append_samples;
      std::string algorithm;
      std::string engine;

      stan_csv_metadata()
        : stan_version_major(0), stan_version_minor(0), stan_version_patch(0),
          model(""), data(""), init(""),
          chain_id(1), seed(0), random_seed(false),
          num_samples(0), num_warmup(0), save_warmup(false), thin(0),
          append_samples(false),
          algorithm(""), engine("") {}
    };

    struct stan_csv_adaptation {
      double step_size;
      Eigen::MatrixXd metric;
      
      stan_csv_adaptation() 
        : step_size(0), metric(0,0) {}
    };
    
    struct stan_csv_timing {
      double warmup;
      double sampling;
      
      stan_csv_timing() 
        : warmup(0), sampling(0) {}
    };

    struct stan_csv {
      stan_csv_metadata metadata;
      Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
      stan_csv_adaptation adaptation;
      Eigen::MatrixXd samples;
      stan_csv_timing timing;
    };

    /**
     * Reads from a Stan output csv file.
     */
    class stan_csv_reader {
    
    public:
      
      stan_csv_reader() {}
      ~stan_csv_reader() {}

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
        
        std::string name;
        std::string value;

        while (ss.good()) {
          
          ss >> comment;
          std::getline(ss, lhs);
          
          size_t equal = lhs.find("=");
          if (equal != std::string::npos) {
            name = lhs.substr(0, equal);
            boost::trim(name);
            value = lhs.substr(equal + 2, lhs.size());
            boost::replace_first(value, " (Default)", "");
          } else {
            continue;
          }
          
          if (name.compare("stan_version_major") == 0) {
            metadata.stan_version_major = boost::lexical_cast<int>(value);
          } else if (name.compare("stan_version_minor") == 0) {
            metadata.stan_version_minor = boost::lexical_cast<int>(value);
          } else if (name.compare("stan_version_patch") == 0) {
            metadata.stan_version_patch = boost::lexical_cast<int>(value);
          } else if (name.compare("model") == 0) {
            metadata.model = value;
          } else if (name.compare("num_samples") == 0) {
            metadata.num_samples = boost::lexical_cast<int>(value);
          } else if (name.compare("num_warmup") == 0) {
            metadata.num_warmup = boost::lexical_cast<int>(value);
          } else if (name.compare("save_warmup") == 0) {
            metadata.save_warmup = boost::lexical_cast<bool>(value);
          } else if (name.compare("thin") == 0) {
            metadata.thin = boost::lexical_cast<int>(value);
          } else if (name.compare("chain_id") == 0) {
            metadata.chain_id = boost::lexical_cast<int>(value);
          } else if (name.compare("data") == 0) {
            metadata.data = value;
          } else if (name.compare("init") == 0) {
            metadata.init = value;
            boost::trim(metadata.init);
          } else if (name.compare("seed") == 0) {
            metadata.seed = boost::lexical_cast<unsigned int>(value);
            metadata.random_seed = false;
          } else if (name.compare("append_samples") == 0) {
            metadata.append_samples = boost::lexical_cast<bool>(value);
          } else if (name.compare("algorithm") == 0) {
            metadata.algorithm = value;
          } else if (name.compare("engine") == 0) {
            metadata.engine = value;
          }
          
        }
        
        if (ss.good() == true)
          return false;
        
        return true;
      
      } // read_metadata
  
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
    
        // Skip first two lines
        std::getline(ss, line);
        
        // Stepsize
        std::getline(ss, line, '=');
        boost::trim(line);
        ss >> adaptation.step_size;
        
        // Metric parameters
        std::getline(ss, line);
        std::getline(ss, line);
        std::getline(ss, line);
        
        int rows = lines - 3;
        int cols = std::count(line.begin(), line.end(), ',') + 1;
        adaptation.metric.resize(rows, cols);
       
        for (int row = 0; row < rows; row++) {
          
          std::stringstream line_ss;
          line_ss.str(line);
          line_ss >> comment;
          
          for (int col = 0; col < cols; col++) {
            std::string token;
            std::getline(line_ss, token, ',');
            boost::trim(token);
            adaptation.metric(row, col) = boost::lexical_cast<double>(token);
          }
          
          std::getline(ss, line); // Read in next line
          
        }
            
        if (ss.good())
          return false;
        else
          return true;

      }
      
      static bool read_samples(std::istream& in, Eigen::MatrixXd& samples, stan_csv_timing& timing) {
        
        std::stringstream ss;
        std::string line;

        int rows = 0;  
        int cols = -1;

        if (in.peek() == '#' || in.good() == false)
          return false;

        while (in.good()) {
          
          bool comment_line = (in.peek() == '#');
          bool empty_line   = (in.peek() == '\n');

          std::getline(in, line);
          
          if (empty_line) continue;
          if (!line.length()) break;
          
          if (comment_line) {

            if (line.find("(Warm-up)") != std::string::npos) {
              int left = 17;
              int right = line.find(" seconds");
              timing.warmup += boost::lexical_cast<double>(line.substr(left, right - left));
            } else if (line.find("(Sampling)") != std::string::npos) {
              int left = 17;
              int right = line.find(" seconds");
              timing.sampling += boost::lexical_cast<double>(line.substr(left, right - left));
            }
            
          }
          else {

            ss << line << '\n';
            
            int current_cols = std::count(line.begin(), line.end(), ',') + 1;
            if (cols == -1) {
              cols = current_cols;
            } else if (cols != current_cols) {
              std::cout << "Error: expected " << cols << " columns, but found " 
                        << current_cols << " instead for row " << rows + 1 << std::endl;
              return false;
            }
            rows++;
            
          }
          
          in.peek();
        
        }
          
        ss.seekg(std::ios_base::beg);

        if (rows > 0) {
          samples.resize(rows, cols);
          for (int row = 0; row < rows; row++) {
            std::getline(ss, line);
            std::stringstream ls(line);
            for (int col = 0; col < cols; col++) {
              std::getline(ls, line, ',');
              boost::trim(line);
              //std::cout << "line: !" << line << "@" << std::endl;
              samples(row, col) = boost::lexical_cast<double>(line);
              //std::cout << "after" << std::endl << std::endl;
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

        data.timing.warmup = 0;
        data.timing.sampling = 0;
        
        if (!read_samples(in, data.samples, data.timing)) {
          std::cout << "Warning: non-fatal error reading samples" << std::endl;
        }
        
        return data;
        
      }
      
    };
    
  } // io
  
} // stan

#endif
