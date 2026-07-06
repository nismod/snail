#ifndef GEOARROW_H
#define GEOARROW_H

#include <pybind11/pybind11.h>

namespace snail {
namespace geoarrow {

/// Register the GeoArrow-based functions and types on a pybind11 module:
/// - split_linestrings: split a GeoArrow array of linestrings along a grid
/// - LineStringArray: GeoArrow linestring array returned by split_linestrings,
///   exposing the Arrow PyCapsule interface (__arrow_c_array__)
void register_module(pybind11::module_ &m);

} // namespace geoarrow
} // namespace snail

#endif // GEOARROW_H
