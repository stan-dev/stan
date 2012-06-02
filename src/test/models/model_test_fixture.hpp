#ifndef __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__
#define __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdio.h>

#include <stan/mcmc/chains.hpp>

namespace testing {
  
  /** 
   * Model_Test_Fixture is a test fixture for google test
   * to aid in running models.
   *
   * Derived classes must define:
   *   static std::vector<std::string> get_model_path()
   * 
   * Template parameters:
   *   bool has_data: indicates whether the model has data
   *   size_t chains: number of chains to run
   */
  template <class Derived,
            bool has_data = false>
  class Model_Test_Fixture : public ::testing::Test {
  
  protected:
    static char path_separator;
    static std::string base_name;
    static const size_t chains = 2;

    static void SetUpTestCase() {
      set_path_separator();
      set_base_name(get_model_path());
    }

    static void set_path_separator() {
      FILE *in;
      if(!(in = popen("make path_separator --no-print-directory", "r")))
        throw std::runtime_error("\"make path_separator\" has failed.");
      path_separator = fgetc(in);
      pclose(in);
    }

    static void
    set_base_name(const std::vector<std::string> model_path) {
      base_name.clear();
      if (model_path.size() < 0) 
        return;
      base_name.append(model_path[0]);
      for (size_t i = 1; i < model_path.size(); i++) {
        base_name.append(1, path_separator);
        base_name.append(model_path[i]);
      }
    }

    static std::string get_command(size_t chain) {
      std::stringstream command;
      command << base_name;
      command << " --samples=" << base_name << "." << chain << ".csv";
      //if (Model_Test_Fixture::has_data) {
      //command << " --data=" << base_name << ".Rdata";
      //}
      return command.str();
    }
    
    static void run() {
      for (size_t chain = 0; chain < chains; chain++) {
        std::string command = get_command(chain);
        EXPECT_EQ(0, system(command.c_str()))
          << "Can not execute command: " << command << std::endl;
      }
    }


  public:
    /** 
     * Return the path to the model (without the extension) as
     * a vector.
     * 
     * @return the path to the model
     */
    static std::vector<std::string> get_model_path() {
      return Derived::get_model_path();
    }
    
  };
  
  template<class Derived, bool has_data> 
  char Model_Test_Fixture<Derived, has_data>::path_separator;
  
  template<class Derived, has_data> 
  std::string Model_Test_Fixture<Derived, has_data>::base_name;
}
#endif
