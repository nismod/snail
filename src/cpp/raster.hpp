#ifndef RASTER_H
#define RASTER_H

#include <vector>
#include <string>

#include "utils.hpp"
#include "geom.hpp"

// Structure defining an ESRI Ascii raster...
struct Ascii{
  int                 ncols;      // number of columns in the Ascii Raster
  int                 nrows;      // number of rows in the Ascii Raster
  double              xll;        // xl corner of the Ascii Raster
  double              yll;        // yll corner of the Ascii Raster
  double              cellsize;   // x, y cell dimension of the incoming data
  double              nodata;     // value taken as "no data"
  std::vector<double> data;       // 1D vector of doubles storing the data in the Ascii Raster
  int                 numCells;   // For convenience, store the number of cells (calculated)

  // Read and interpret a line in the ascii header...
  void readHeaderLine(const std::string line) {
    std::vector<std::string> key_value = utils::readLine(line, ' ');

    // Interpret the key-value pair from the header line...
    std::string key   = utils::lower_case(key_value.at(0));
    std::string value = key_value.at(1);

    // Check the incoming strings against the known header...
    if(key == "ncols")
      ncols = std::stoi(value);
    else if(key == "nrows")
      nrows = std::stoi(value);
    else if(key == "xllcorner")
      xll = std::stod(value);
    else if(key == "yllcorner")
      yll = std::stod(value);
    else if(key == "cellsize")
      cellsize = std::stod(value);
    else if(key == "nodata_value")
      nodata = std::stoi(value); // nodata is taken to be an int, not a float...
  }

  // Helper method to calculate hashed index in raster...
  int cellIndex(const geometry::Vec2<double> p) const {
    geometry::Vec2<long double> offset((p.x - xll) / cellsize, (p.y - yll) / cellsize);
    return floor(offset.x) + floor(offset.y)*ncols;
  }

  // Helper method to recover i, j index in raster...
  geometry::Vec2<int> cellIndices(const geometry::Vec2<double> p) const {
    geometry::Vec2<double> offset((p.x - xll) / cellsize, (p.y - yll) / cellsize);
    return geometry::Vec2<int>(floor(offset.x), floor(offset.y));
  }

  // Helper method to calculate the position of a point in a cell...
  geometry::Vec2<double> offsetInCell(const geometry::Vec2<double> p) const {
    // Retrieve the indices of the cell...
    geometry::Vec2<int> cell_offset = cellIndices(p);

    // Calculate the LL coordinates of the cell...
    double x0 = xll + cell_offset.x*cellsize;
    double y0 = yll + cell_offset.y*cellsize;

    // Return the distance between the two points...
    return geometry::Vec2<double>(p.x - x0, p.y - y0);
  }
  
  // Method to calculate the points at which a line-segment intersects the Ascii grid lines / graticules...
  std::vector<geometry::Vec2<double>> findIntersections(const geometry::Line2<double> line) const {
    // First, calculate the run and rise of the line...
    double run  = (line.end.x - line.start.x);
    double rise = (line.end.y - line.start.y);

    // Calculate the length of the line segment being passed in for testing...
    double length = line.length();

    // Work out where the start-point of the line falls in the cell...
    geometry::Vec2<double> delta = offsetInCell(line.start);

    // Determine which cell boundaries the line will cross (N or S, E or W)...
    int north = rise >= 0 ? 2 : 0;
    int east  = run  >= 0 ? 2 : 0;

    // And work out the appropriate delta from the start of the line to the crossing boundary...
    double dN = north ? cellsize - delta.y : -delta.y;
    double dE = east  ? cellsize - delta.x : -delta.x;

    // Calculate the positions at which the line crosses the grid graticules...
    geometry::Vec2<double> pN(dN*run/rise, dN);
    geometry::Vec2<double> pE(dE, dE*rise/run);

    // Create a vector of grid / graticule crossings to return to the caller...
    std::vector<geometry::Vec2<double>> crossings;

    // Append the start point of the line to the vector of crossings...
    crossings.push_back(line.start);

    // As long as there is a crossing point BEFORE the end of the line, we can keep looping...
    while(pE.length() < length || pN.length() < length){
      // Add the closest crossing point to the vector of grid / graticule crossings...
      if(pE.length() < pN.length()){
	crossings.push_back(line.start + pE);
	// Update the distance to the next graticule...
	dE += double(east-1)*cellsize;
	// Calculate the position of the crossing point on the next grid / graticule line...
	pE = geometry::Vec2<double>(dE, dE*rise/run);
      }else{
	crossings.push_back(line.start + pN);
	// Update the distance to the next graticule...
	dN += double(north-1)*cellsize;
	// Calculate the position of the crossing point on the next grid / graticule line...
	pN = geometry::Vec2<double>(dN*run/rise, dN);
      }
    }

    // Return the vector of grid / graticule crossing points that exist bbetween the start and end of the line...
    return crossings;
  }

  // Ascii grid Constructor...
  Ascii(const std::string filename){
    // Test if the ascii raster exists...
    if(!utils::exists(filename))
      utils::Exception("The Ascii raster you are trying to open does not exist (" + filename + ")");

    // Open the nominated file...
    std::ifstream infile;
    infile.open(filename);

    // Create somewhere to read the data...
    std::string line;

    // Read the header, process the data...
    for(int i=0; i<6; i++){
      std::getline(infile, line);
      readHeaderLine(line);
    }

    // Reserve some space for the data that is in the file...
    data.resize(nrows*ncols);

    // Read the rest of the data (NOTE: The file is being read in top-bottom in line with the file format)...
    for(int j=nrows-1; j>0; j--){
      std::getline(infile, line);
      // Extract the data from the line...
      std::vector<std::string> words = utils::readLine(line,' ');
      // Stick the data on the tab...
      int i=0;
      for(auto word : words){
	// Calculate the index of the data, populate it...
	data.at(i + j*ncols) = std::max(0.0,std::stod(word));
	i++;
      }
    }

    // For ease of testing later on, store the total number of cells in the grid...
    numCells = data.size() - 1;
  }


};


#endif // RASTER_H
