#ifndef __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__
#define __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__

#include <gtest/gtest.h>
#include <stan/mcmc/chains.hpp>


/** 
 * Model_Test_Fixture is a test fixture for google test
 * to aid in running models.
 *
 * Derived classes must define:
 *   static std::vector<std::string> get_model_path()
 *   static bool has_data()
 *   - indicates whether the model has data
 */
template <class Derived>
class Model_Test_Fixture : public ::testing::Test {
  
public:
  static char path_separator;
  static std::string model_path;
  static stan::mcmc::chains<> *chains;
  static size_t num_chains;

  /** 
   * SetUpTestCase() called by google test once
   * a test case.
   *
   * Sets:
   *  - path_separator: '\' for Windows, '/' for all else.
   *  - base_name:      location of model without ".stan"
   *  - chains:         creates a chains object by reading
   *                    the header of a csv file
   */
  static void SetUpTestCase() {
    path_separator = get_path_separator();
    model_path = convert_model_path(get_model_path());
      
    chains = create_chains();
  }
    
  /** 
   * TearDownTestCase() called by google test once
   * a test case.
   *
   * Deletes chains. 
   */
  static void TearDownTestCase() {
    delete chains;
  }

  /** 
   * Gets the path separator for the OS.
   * 
   * @return '\' for Windows, '/' otherwise.
   */
  static char get_path_separator() {
    char c;
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    c = fgetc(in);
    pclose(in);
    return c;
  }

  /** 
   * Returns the path as a string with the appropriate
   * path separator.
   * 
   * @param model_path vector of strings representing path to the model
   * 
   * @return the string representation of the path with the appropriate
   *    path separator.
   */
  static std::string
  convert_model_path(const std::vector<std::string> model_path) {
    std::string path;
    if (model_path.size() > 0) {
      path.append(model_path[0]);
      for (size_t i = 1; i < model_path.size(); i++) {
        path.append(1, path_separator);
        path.append(model_path[i]);
      }
    }
    return path;
  }

  /** 
   * Returns the csv file for the chain number.
   * 
   * @param chain the chain number
   * 
   * @return the file location of the csv file
   */
  static std::string get_csv_file(size_t chain) {
    std::stringstream csv_file;
    csv_file << model_path << "." << chain << ".csv";
    return csv_file.str();
  }

  /** 
   * Returns the command to run in a shell on the OS.
   * 
   * @param chain the chain number 
   * 
   * @return a string command that can be run from a shell.
   */
  static std::string get_command(size_t chain) {
    std::stringstream command;
    command << model_path;
    command << " --samples=" << get_csv_file(chain);
    if (has_data()) {
      command << " --data=" << model_path << ".Rdata";
    }
    return command.str();
  }


  /** 
   * Populates the chains object with data from csv files.
   */
  static void populate_chains() {
    if (chains->num_kept_samples() == 0U) {
      for (size_t chain = 0U; chain < num_chains; chain++) {
        stan::mcmc::add_chain(*chains, chain, get_csv_file(chain), 2U);
      }
    }
  }
  
    
  /** 
   * Runs the model num_chains times.
   * Populates the chains object after running the model.
   */
  static void run_model() {
    for (size_t chain = 0; chain < num_chains; chain++) {
      std::string command = get_command(chain);
      EXPECT_EQ(0, system(command.c_str()))
        << "Can not execute command: " << command << std::endl;
    }
    populate_chains();
  }
    
  /** 
   * Creates a chains object.
   *
   * Runs the model for 0 iterations to read the
   * names and dimensions of the parameters.
   * 
   * @return An initialized chains object.
   */
  static stan::mcmc::chains<>* create_chains() {
    std::string command = get_command(0U);
    command += " --iter=0";
    EXPECT_EQ(0, system(command.c_str()))
      << "Can not build header using: " << command << std::endl;
      
    std::vector<std::string> names;
    std::vector<std::vector<size_t> > dimss;
    stan::mcmc::read_variables(get_csv_file(0U), 2,
                               names, dimss);
      
    return (new stan::mcmc::chains<>(num_chains, names, dimss));
  }


  /** 
   * Return the path to the model (without the extension) as
   * a vector.
   * 
   * @return the path to the model
   */
  static std::vector<std::string> get_model_path() {
    return Derived::get_model_path();
  }
  
  /**
   * Return true if the model has data.
   *
   * @return true if the model has data;
   *         false otherwise.
   */
  static bool has_data() {
    return Derived::has_data();
  }
    
};
  
template<class Derived> 
char Model_Test_Fixture<Derived>::path_separator;

template<class Derived> 
stan::mcmc::chains<> *Model_Test_Fixture<Derived>::chains;

template<class Derived>
size_t Model_Test_Fixture<Derived>::num_chains = 2;

template<class Derived>
std::string Model_Test_Fixture<Derived>::model_path;



TYPED_TEST_CASE_P(Model_Test_Fixture);

TYPED_TEST_P(Model_Test_Fixture, RunModel) {
  TypeParam::run_model();
}

REGISTER_TYPED_TEST_CASE_P(Model_Test_Fixture,
			   RunModel);

#endif
