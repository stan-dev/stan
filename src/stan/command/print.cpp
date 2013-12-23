#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ios>
#include <stan/mcmc/chains.hpp>
#include <stan/command/print.hpp>

/**
 * The Stan print function.
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * 
 * @return 0 for success, 
 *         non-zero otherwise
 */
int main(int argc, const char* argv[]) {
  
  if (argc == 1) {
    print_usage();
    return 0;
  }
  
  // Parse any arguments specifying filenames
  std::ifstream ifstream;
  std::vector<std::string> filenames;
  
  for (int i = 1; i < argc; i++) {
    
    if (std::string(argv[i]).find("--autocorr=") != std::string::npos)
      continue;
    
    if (std::string(argv[i]).find("--sig_figs=") != std::string::npos)
      continue;
    
    if (std::string("--help") == std::string(argv[i])) {
      print_usage();
      return 0;
    }
    
    ifstream.open(argv[i]);
    
    if (ifstream.good()) {
      filenames.push_back(argv[i]);
      ifstream.close();
    } else {
      std::cout << "File " << argv[i] << " not found" << std::endl;
    }
    
  }
  
  if (!filenames.size()) {
    std::cout << "No valid input files, exiting." << std::endl;
    return 0;
  }
  
  // Parse specified files
  Eigen::VectorXd warmup_times(filenames.size());
  Eigen::VectorXd sampling_times(filenames.size());
  
  Eigen::VectorXi thin(filenames.size());
  
  ifstream.open(filenames[0].c_str());
  
  stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
  warmup_times(0) = stan_csv.timing.warmup;
  sampling_times(0) = stan_csv.timing.sampling;
  
  stan::mcmc::chains<> chains(stan_csv);
  ifstream.close();
  
  thin(0) = stan_csv.metadata.thin;
  
  for (std::vector<std::string>::size_type chain = 1; 
       chain < filenames.size(); chain++) {
    ifstream.open(filenames[chain].c_str());
    stan_csv = stan::io::stan_csv_reader::parse(ifstream);
    chains.add(stan_csv);
    ifstream.close();
    thin(chain) = stan_csv.metadata.thin;
    
    warmup_times(chain) = stan_csv.timing.warmup;
    sampling_times(chain) = stan_csv.timing.sampling;
    
  }
  
  double total_warmup_time = warmup_times.sum();
  double total_sampling_time = sampling_times.sum();

  // Compute largest variable name length
  const int skip = 0;
  std::string model_name = stan_csv.metadata.model;
  size_t max_name_length = 0;
  for (int i = skip; i < chains.num_params(); i++) 
    if (chains.param_name(i).length() > max_name_length)
      max_name_length = chains.param_name(i).length();
  for (int i = 0; i < 2; i++) 
    if (chains.param_name(i).length() > max_name_length)
      max_name_length = chains.param_name(i).length();


  // Prepare values
  int n = 9;
  
  Eigen::MatrixXd values(chains.num_params(), n);
  values.setZero();
  Eigen::VectorXd probs(3);
  probs << 0.05, 0.5, 0.95;
  
  for (int i = 0; i < chains.num_params(); i++) {
    double sd = chains.sd(i);
    double n_eff = chains.effective_sample_size(i);
    values(i,0) = chains.mean(i);
    values(i,1) = sd / sqrt(n_eff);
    values(i,2) = sd;
    Eigen::VectorXd quantiles = chains.quantiles(i,probs);
    for (int j = 0; j < 3; j++)
      values(i,3+j) = quantiles(j);
    values(i,6) = n_eff;
    values(i,7) = n_eff / total_sampling_time;
    values(i,8) = chains.split_potential_scale_reduction(i);
  }
  
  // Prepare header
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> headers(n);
  headers << 
    "Mean", "MCSE", "StdDev",
    "5%", "50%", "95%", 
    "N_Eff", "N_Eff/s", "R_hat";
  
  // Set sig figs
  Eigen::VectorXi column_sig_figs(n);
  
  int sig_figs = 2;
  
  for (int k = 1; k < argc; k++)
    if (std::string(argv[k]).find("--sig_figs=") != std::string::npos)
      sig_figs = atoi(std::string(argv[k]).substr(11).c_str());
  
  // Compute column widths
  Eigen::VectorXi column_widths(n);
  Eigen::Matrix<std::ios_base::fmtflags, Eigen::Dynamic, 1> formats(n);
  column_widths = calculate_column_widths(values, headers, sig_figs, formats);
  
  // Initial output
  std::cout << "Inference for Stan model: " << model_name << std::endl
            << chains.num_chains() << " chains: each with iter=(" << chains.num_kept_samples(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << chains.num_kept_samples(chain);
  std::cout << ")";
  
  // Timing output
  std::cout << "; warmup=(" << chains.warmup(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << chains.warmup(chain);
  std::cout << ")";
                          
  std::cout << "; thin=(" << thin(0);
  
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << thin(chain);
  std::cout << ")";
                          
  std::cout << "; " << chains.num_samples() << " iterations saved."
            << std::endl << std::endl;

  std::string warmup_unit = "seconds";
  
  if (total_warmup_time / 3600 > 1) {
    total_warmup_time /= 3600;
    warmup_unit = "hours";
  } else if (total_warmup_time / 60 > 1) {
    total_warmup_time /= 60;
    warmup_unit = "minutes";
  }
  
  std::cout << "Warmup took ("
            << std::fixed
            << std::setprecision(compute_precision(warmup_times(0), sig_figs, false))
            << warmup_times(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << ", " << std::fixed
              << std::setprecision(compute_precision(warmup_times(chain), sig_figs, false))
              << warmup_times(chain);
  std::cout << ") seconds, ";
  std::cout << std::fixed
            << std::setprecision(compute_precision(total_warmup_time, sig_figs, false))
            << total_warmup_time << " " << warmup_unit << " total" << std::endl;

  std::string sampling_unit = "seconds";
  
  if (total_sampling_time / 3600 > 1) {
    total_sampling_time /= 3600;
    sampling_unit = "hours";
  } else if (total_sampling_time / 60 > 1) {
    total_sampling_time /= 60;
    sampling_unit = "minutes";
  }

  std::cout << "Sampling took ("
            << std::fixed
            << std::setprecision(compute_precision(sampling_times(0), sig_figs, false))
            << sampling_times(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << ", " << std::fixed
              << std::setprecision(compute_precision(sampling_times(chain), sig_figs, false))
              << sampling_times(chain);
  std::cout << ") seconds, ";
  std::cout << std::fixed
            << std::setprecision(compute_precision(total_sampling_time, sig_figs, false))
            << total_sampling_time << " " << sampling_unit << " total" << std::endl;
  std::cout << std::endl;

  // Header output
  std::cout << std::setw(max_name_length + 1) << "";
  for (int i = 0; i < n; i++) {
    std::cout << std::setw(column_widths(i)) << headers(i);
  }
  std::cout << std::endl;
  
  // Value output
  for (int i = skip; i < chains.num_params(); i++) {
    if (!is_matrix(chains.param_name(i))) {
      std::cout << std::setw(max_name_length + 1) << std::left << chains.param_name(i);
      std::cout << std::right;
      for (int j = 0; j < n; j++) {
        std::cout.setf(formats(j), std::ios::floatfield);
        std::cout << std::setprecision(
                                       compute_precision(values(i,j), sig_figs, formats(j) == std::ios_base::scientific))
                  << std::setw(column_widths(j)) << values(i, j);
      }
      std::cout << std::endl;
    } else {
      std::vector<int> dims = dimensions(chains, i);
      std::vector<int> index(dims.size(), 1);
      int max = 1;
      for (size_t j = 0; j < dims.size(); j++)
        max *= dims[j];
      
      for (int k = 0; k < max; k++) {
        int param_index = i + matrix_index(index, dims);
        std::cout << std::setw(max_name_length + 1) << std::left 
                  << chains.param_name(param_index);
        std::cout << std::right;
        for (int j = 0; j < n; j++) {
          std::cout.setf(formats(j), std::ios::floatfield);
          std::cout 
            << std::setprecision(compute_precision(values(param_index,j), 
                                                   sig_figs, 
                                                   formats(j) == std::ios_base::scientific))
            << std::setw(column_widths(j)) << values(param_index, j);
        }
        std::cout << std::endl;
        if (k < max-1)
          next_index(index, dims);
      }
      i += max-1;
    }
  }

  /// Footer output
  std::cout << std::endl;
  std::cout << "Samples were drawn using " << stan_csv.metadata.algorithm
            << " with " << stan_csv.metadata.engine << "." << std::endl
            << "For each parameter, N_Eff is a crude measure of effective sample size," << std::endl
            << "and R_hat is the potential scale reduction factor on split chains (at " << std::endl
            << "convergence, R_hat=1)." << std::endl
            << std::endl;
  
  // Print autocorrelation, if desired
  for (int k = 1; k < argc; k++) {
    
    if (std::string(argv[k]).find("--autocorr=") != std::string::npos) {
      
      const int c = atoi(std::string(argv[k]).substr(11).c_str());
      
      if (c < 0 || c >= chains.num_chains()) {
        std::cout << "Bad chain index " << c << ", aborting autocorrelation display." << std::endl;
        break;
      }
      
      Eigen::MatrixXd autocorr(chains.num_params(), chains.num_samples(c));
      
      for (int i = 0; i < chains.num_params(); i++) {
        autocorr.row(i) = chains.autocorrelation(c, i);
      }
      
      // Format and print header
      std::cout << "Displaying the autocorrelations for chain " << c << ":" << std::endl;
      std::cout << std::endl;
      
      const int n_autocorr = autocorr.row(0).size();
      
      int lag_width = 1;
      int number = n_autocorr; 
      while ( number != 0) { number /= 10; lag_width++; }

      std::cout << std::setw(lag_width > 4 ? lag_width : 4) << "Lag";
      for (int i = 0; i < chains.num_params(); ++i) {
        std::cout << std::setw(max_name_length + 1) << std::right << chains.param_name(i);
      }
      std::cout << std::endl;

      // Print body  
      for (int n = 0; n < n_autocorr; ++n) {
        std::cout << std::setw(lag_width) << std::right << n;
        for (int i = 0; i < chains.num_params(); ++i) {
          std::cout << std::setw(max_name_length + 1) << std::right << autocorr(i, n);
        }
        std::cout << std::endl;
      }
  
    }
        
  }
      
  return 0;
        
}



