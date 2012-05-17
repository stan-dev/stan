#include <gtest/gtest.h>
#include <stdexcept>

class Models_BasicDistributions_SortedPair : public ::testing::Test {
protected:
  virtual void SetUp() {
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    path_separator += fgetc(in);
    pclose(in);
    
    model.append("models").append(path_separator);
    model.append("basic_distributions").append(path_separator);
    model.append("sorted_pair");

    output1 = model + "1.csv";
    output2 = model + "2.csv";
  }
  std::string path_separator;
  std::string model;
  std::string output1;
  std::string output2;
};

TEST_F(Models_BasicDistributions_SortedPair,RunModel) {
  std::string command;
  command = model;
  command += " --samples=";
  command += output1;
  EXPECT_EQ(0, system(command.c_str())) 
    << "Can not execute command: " << command << std::endl;
            
  
  command = model;
  command += " --samples=";
  command += output2;
  EXPECT_EQ(0, system(command.c_str()))
    << "Can not execute command: " << command << std::endl;
}
