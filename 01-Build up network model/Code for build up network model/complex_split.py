""""
Taken from https://github.com/shapely/shapely/issues/1068#issuecomment-770296614

Split a complex linestring using shapely.

Inspired by https://github.com/Toblerity/Shapely/issues/1068
"""
from shapely.geometry import LineString, GeometryCollection, Point, Polygon
from shapely.ops import split, snap


def complex_split(geom: LineString, splitter):
    """Split a complex linestring by another geometry without splitting at
    self-intersection points.

    Parameters
    ----------
    geom : LineString
        An optionally complex LineString.
    splitter : Geometry
        A geometry to split by.

    Warnings
    --------
    A known vulnerability is where the splitter intersects the complex
    linestring at one of the self-intersecting points of the linestring.
    In this case, only one the first path through the self-intersection
    will be split.

    Examples
    --------
    >>> complex_line_string = LineString([(0, 0), (1, 1), (1, 0), (0, 1)])
    >>> splitter = LineString([(0, 0.5), (0.5, 1)])
    >>> complex_split(complex_line_string, splitter).wkt
    'GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1, 1 0, 0.25 0.75), LINESTRING (0.25 0.75, 0 1))'

    Return
    ------
    GeometryCollection
        A collection of the geometries resulting from the split.
    """
    if geom.is_simple:
        return split(geom, splitter)

    if isinstance(splitter, Polygon):
        splitter = splitter.exterior

    # Ensure that intersection exists and is zero dimensional.
    relate_str = geom.relate(splitter)
    if relate_str[0] == '1':
        raise ValueError('Cannot split LineString by a geometry which intersects a '
                         'continuous portion of the LineString.')
    if not (relate_str[0] == '0' or relate_str[1] == '0'):
        return GeometryCollection((geom,))

    intersection_points = geom.intersection(splitter)
    # This only inserts the point at the first pass of a self-intersection if
    # the point falls on a self-intersection.
    snapped_geom = snap(geom, intersection_points, tolerance=1.0e-12)  # may want to make tolerance a parameter.
    # A solution to the warning in the docstring is to roll your own split method here.
    # The current one in shapely returns early when a point is found to be part of a segment.
    # But if the point was at a self-intersection it could be part of multiple segments.
    return split(snapped_geom, intersection_points)


if __name__ == '__main__':
    complex_line_string = LineString([(0, 0), (1, 1), (1, 0), (0, 1)])
    splitter = LineString([(0, 0.5), (0.5, 1)])

    out = complex_split(complex_line_string, splitter)
    print(out)
    assert len(out) == 2

    # test inserting and splitting at self-intersection
    pt = Point(0.5, 0.5)
    print(f'snap: {snap(complex_line_string, pt, tolerance=1.0e-12)}')
    print(f'split: {split(snap(complex_line_string, pt, tolerance=1.0e-12), pt)}')

