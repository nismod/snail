#ifndef GEOARROW_H
#define GEOARROW_H

#include <pybind11/pybind11.h>

namespace snail {
namespace geoarrow {

/// Register the GeoArrow data exchange functions and types on a pybind11
/// module:
/// - split_linestrings / split_polygons: split a whole GeoArrow array of
///   geometries along a grid, returning GeoArrow pieces
/// - GeoArrowArray: the returned array, exposing the Arrow PyCapsule
///   interface (__arrow_c_array__)
void register_module(pybind11::module_ &m);

} // namespace geoarrow
} // namespace snail

#endif // GEOARROW_H
