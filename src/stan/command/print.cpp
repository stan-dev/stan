#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ios>
#include <stan/mcmc/chains.hpp>


int calculate_size(const Eigen::VectorXd& x, 
                   const std::string& name,
                   const int digits,
                   std::ios_base::fmtflags& format) {
  using std::max;
  using std::ceil;
  using std::log10;
  
  double padding = 0;
  if (digits > 0)
    padding = digits + 1;

  double fixed_size = 0.0;
  if (x.maxCoeff() > 0)
    fixed_size = ceil(log10(x.maxCoeff()+0.001)) + padding;
  if (x.minCoeff() < 0)
    fixed_size = max(fixed_size, ceil(log10(-x.minCoeff()+0.01))+(padding+1));
  format = std::ios_base::fixed;
  if (fixed_size < 7) {
    return max(fixed_size,
               max(name.length(), std::string("-0.0").length())+0.0);
  }

  double scientific_size = 0;
  scientific_size += 4.0;   // "-0.0" has four digits
  scientific_size += 1.0;   // e
  double exponent_size = 0;
  if (x.maxCoeff() > 0)
    exponent_size = ceil(log10(log10(x.maxCoeff())));
  if (x.minCoeff() < 0)
    exponent_size = max(exponent_size,
                        ceil(log10(log10(-x.minCoeff()))));
  scientific_size += fmin(exponent_size, 3);
  format = std::ios_base::scientific;
  return scientific_size;
}

Eigen::VectorXi calculate_sizes(const Eigen::MatrixXd& values, 
                                const Eigen::Matrix<std::string, Eigen::Dynamic, 1>& headers, 
                                const Eigen::VectorXi& digits,
                                Eigen::Matrix<std::ios_base::fmtflags, Eigen::Dynamic, 1>& formats) {
  int n = values.cols();
  Eigen::VectorXi column_lengths(n);
  formats.resize(n);
  for (int i = 0; i < n; i++) {
    column_lengths(i) = calculate_size(values.col(i), headers(i), digits(i), formats(i)) + 1;
  }
  return column_lengths;
}

void print_usage() {
  
  std::cout << "USAGE:  print <filename 1> [<filename 2> ... <filename N>]"
            << std::endl
            << std::endl;
  
  std::cout << "OPTIONS:" << std::endl << std::endl;
  std::cout << "  --autocorr=<chain_index>\tAppend the autocorrelations for the given chain"
            << std::endl
            << std::endl;
  
}


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
  
  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i++) {
    
    if (std::string(argv[i]).find("--autocorr=") != std::string::npos)
      continue;
    
    filenames.push_back(argv[i]);
    
    if (std::string("--help") == std::string(argv[i])) {
      print_usage();
      return 0;
      
    }
  }
  
  Eigen::VectorXi thin(filenames.size());
  
  std::ifstream ifstream;
  ifstream.open(filenames[0].c_str());
  stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
  stan::mcmc::chains<> chains(stan_csv);
  ifstream.close();
  thin(0) = stan_csv.metadata.thin;
  

  for (int chain = 1; chain < filenames.size(); chain++) {
    ifstream.open(filenames[chain].c_str());
    stan_csv = stan::io::stan_csv_reader::parse(ifstream);
    chains.add(stan_csv);
    ifstream.close();
    thin(chain) = stan_csv.metadata.thin;
  }

  // print  
  const int skip = 3;
  std::string model_name = ""; // FIXME: put in model name
  int max_name_length = 0;
  for (int i = skip; i < chains.num_params(); i++) 
    if (chains.param_name(i).length() > max_name_length)
      max_name_length = chains.param_name(i).length();
  for (int i = 0; i < 2; i++) 
    if (chains.param_name(i).length() > max_name_length)
      max_name_length = chains.param_name(i).length();


  Eigen::MatrixXd values(chains.num_params(),10);
  values.setZero();
  Eigen::VectorXd probs(5);
  probs << 0.025, 0.25, 0.5, 0.75, 0.975;
  
  for (int i = 0; i < chains.num_params(); i++) {
    double sd = chains.sd(i);
    double n_eff = chains.effective_sample_size(i);
    values(i,0) = chains.mean(i);
    values(i,1) = sd / sqrt(n_eff);
    values(i,2) = sd;
    Eigen::VectorXd quantiles = chains.quantiles(i,probs);
    for (int j = 0; j < 5; j++)
      values(i,3+j) = quantiles(j);
    values(i,8) = n_eff;
    values(i,9) = chains.split_potential_scale_reduction(i);
  }
  
  int n = 10;
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> headers(n);
  headers << 
    "mean", "se_mean", "sd", 
    "2.5%", "25%", "50%", "75%", "97.5%", 
    "n_eff", "Rhat";
  Eigen::VectorXi digits(n);
  digits.setConstant(1);
  digits(8) = 0;
  
  // Want per row:
  //   scientific vs floating point
  // Want per column:
  //   length
  
  Eigen::VectorXi column_lengths(n);
  // Formats should be a vector of length chains.num_params()
  Eigen::Matrix<std::ios_base::fmtflags, Eigen::Dynamic, 1> formats(n);
  column_lengths = calculate_sizes(values, headers, digits, formats);
  
  std::cout << "Inference for Stan model: " << model_name << std::endl
            << chains.num_chains() << " chains: each with iter=(" << chains.num_kept_samples(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << chains.num_kept_samples(chain);
  std::cout << ")";
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
  
  using std::setprecision;
  using std::setw;

  // header
  std::cout << std::setw(max_name_length+1) << "";
  for (int i = 0; i < n; i++) {
    std::cout << setw(column_lengths(i)) << headers(i);
  }
  std::cout << std::endl;
  // each row
  for (int i = skip; i < chains.num_params(); i++) {
    std::cout << setw(max_name_length+1) << std::left << chains.param_name(i);
    std::cout << std::right;
    for (int j = 0; j < n; j++) {
      std::cout.setf(formats(j), std::ios::floatfield);
      std::cout << setprecision(digits(j)) << setw(column_lengths(j)) << values(i,j);
    }
    std::cout << std::endl;
  }
  // lp__, treedepth__
  for (int i = 0; i < 2; i++) {
    std::cout << setw(max_name_length+1) << std::left << chains.param_name(i);
    std::cout << std::right;
    for (int j = 0; j < n; j++) {
      std::cout.setf(formats(j), std::ios::floatfield);
      std::cout << setprecision(digits(j)) << setw(column_lengths(j)) << values(i,j);
    }
    std::cout << std::endl;
  }
    
  std::cout << std::endl;
  std::cout << "Samples were drawn using " << stan_csv.adaptation.sampler << "." << std::endl
            << "For each parameter, n_eff is a crude measure of effective sample size," << std::endl
            << "and Rhat is the potential scale reduction factor on split chains (at " << std::endl
            << "convergence, Rhat=1)." << std::endl
            << std::endl;
  
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

      std::cout << setw(lag_width > 4 ? lag_width : 4) << "Lag";
      for (int i = 0; i < chains.num_params(); ++i) {
        std::cout << setw(max_name_length + 1) << std::right << chains.param_name(i);
      }
      std::cout << std::endl;

      // Print body  
      for (int n = 0; n < n_autocorr; ++n) {
        std::cout << setw(lag_width) << std::right << n;
        for (int i = 0; i < chains.num_params(); ++i) {
          std::cout << setw(max_name_length + 1) << std::right << autocorr(i, n);
        }
        std::cout << std::endl;
      }
  
    }
        
  }
      
  return 0;
        
}



