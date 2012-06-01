#ifndef __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__
#define __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>


namespace testing {

  class Model_Test_Fixture : public ::testing::Test {
  protected:
    static std::string path_separator;
    
    static void SetUpTestCase() {
      FILE *in;
      if(!(in = popen("make path_separator --no-print-directory", "r")))
        throw std::runtime_error("\"make path_separator\" has failed.");
      path_separator = "";
      path_separator += fgetc(in);
      pclose(in);
    }

  };
  
  std::string Model_Test_Fixture::path_separator;
}
#endif
