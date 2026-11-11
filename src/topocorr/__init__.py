"""topocorr: terrain illumination correction and illumination-artefact tests.

topocorr works on Landsat 5-9 Collection 2 Level 2 and Sentinel-2 L2A collections in
Google Earth Engine. It computes the local solar illumination of every acquisition from
the DEM and the image's sun angles, flags shadowed and steep pixels, and applies the
standard terrain corrections.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
