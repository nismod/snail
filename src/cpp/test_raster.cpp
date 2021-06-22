#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "raster.h"

int main(int argc, char** argv){
  std::string ascii_file       = std::string(argv[1]);
  Ascii flood_depth(ascii_file);
}
