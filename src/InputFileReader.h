/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef INPUT_FILE_READER_H
#define INPUT_FILE_READER_H

#if GOMC_LIB_MPI
#include <mpi.h>
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

class InputFileReader {
private:
  std::ifstream fs;
  std::vector<std::string> &split(const std::string &s, char delim,
                                  std::vector<std::string> &elems);

public:
  bool readNextLine(std::vector<std::string> &str);
  void Open(std::string fileName);
  void Test(std::string fileName);
  InputFileReader(std::string fileName);
  InputFileReader() {}
  ~InputFileReader();
};

#endif /*INPUT_FILE_READER_H*/
