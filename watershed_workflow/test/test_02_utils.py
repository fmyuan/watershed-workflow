import pytest
import numpy as np
import shapely.geometry

import watershed_workflow.utils
from watershed_workflow.test.shapes import two_boxes


def test_close():
    # points
    p0 = shapely.geometry.Point((0., 0.))
    p1 = shapely.geometry.Point((0., 0.))
    assert (watershed_workflow.utils.isClose(p0, p1))

    p1 = shapely.geometry.Point((0., 0.000001))
    assert (not watershed_workflow.utils.isClose(p0, p1))

    p1 = shapely.geometry.Point((0., 1.e-8))
    assert (watershed_workflow.utils.isClose(p0, p1))

    # point and tuples
    assert (watershed_workflow.utils.isClose(p0, (0, 1.e-8)))
    assert (watershed_workflow.utils.isClose((0, 0), (0, 1.e-8)))

    # fails
    assert (not watershed_workflow.utils.isClose((0, 0, 0), (0, 0)))

    # lineseg
    l0 = shapely.geometry.LineString([(0, 0), (1, 0), (2, 0)])
    l1 = shapely.geometry.LineString([(0, 0.001), (1, 0), (2, 0)])
    l2 = shapely.geometry.LineString([(0, 0.001), (2, 0)])
    assert (watershed_workflow.utils.isClose(l0, l1, 0.01))
    assert (not watershed_workflow.utils.isClose(l0, l1, 0.0001))
    assert (not watershed_workflow.utils.isClose(l0, l2, 100))

    # polygon
    p0 = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p1 = shapely.geometry.Polygon([(1, 0.001), (1.001, 1), (0, 1), (0, 0)])
    p2 = shapely.geometry.Polygon([(0, 0), (0, 1), (1.001, 1), (1, 0.001)])
    assert (watershed_workflow.utils.isClose(p0, p1, 0.01))
    assert (not watershed_workflow.utils.isClose(p0, p1, 0.0001))
    assert (watershed_workflow.utils.isClose(p0, p2, 0.01))
    assert (not watershed_workflow.utils.isClose(p0, p2, 0.0001))


def test_contains():
    coords = np.array([[1.03425, 0.0013], [0.0035, 1.03523], [-1.09824, 0.0033], [0.0012,
                                                                                  -1.04856]])
    linecoords = np.array([[0.1394, 0.0492], [3.1415, 1.1394]])

    def wiggle(coords):
        random = np.random.random((len(coords), 2))
        random = 2 * (random-.5) * .01
        return coords + random

    contains = []
    for i in range(100):
        newc = wiggle(coords)
        newl = wiggle(linecoords)
        shp = shapely.geometry.Polygon(newc)
        line = shapely.geometry.LineString(newl)
        lines, _ = watershed_workflow.utils.cut(line, shp.boundary)

        contains.append(watershed_workflow.utils.contains(shp, lines[0]))

    print("Contains % = ", sum(1 for i in contains if i) / 100.0)
    assert (all(contains))


def test_cut_point_not_there():
    line = shapely.geometry.LineString([(0, 0), (1, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines, _ = watershed_workflow.utils.cut(line, cut)
    assert (len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print(l1.coords[:])
    print(l2.coords[:])
    assert (l1 == shapely.geometry.LineString([(0, 0), (0.5, 0)]))
    assert (l2 == shapely.geometry.LineString([(0.5, 0), (1, 0)]))


def test_cut_point_there():
    line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines, _ = watershed_workflow.utils.cut(line, cut)
    assert (len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print(l1.coords[:])
    print(l2.coords[:])
    assert (l1 == shapely.geometry.LineString([(0, 0), (0.5, 0)]))
    assert (l2 == shapely.geometry.LineString([(0.5, 0), (1, 0)]))


#
# NOTE: cut no longer snaps!
#

# def test_cut_point_nearly_there_after():
#     line = shapely.geometry.LineString([(0, 0), (0.50001, 0), (1, 0)])
#     cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
#     lines, _ = watershed_workflow.utils.cut(line, cut)
#     assert (len(lines) == 2)
#     l1 = lines[0]
#     l2 = lines[1]
#     print(l1.coords[:])
#     print(l2.coords[:])
#     assert (l1 == shapely.geometry.LineString([(0, 0), (0.5, 0)]))
#     assert (l2 == shapely.geometry.LineString([(0.5, 0), (1, 0)]))


# def test_cut_point_nearly_there_before():
#     line = shapely.geometry.LineString([(0, 0), (0.49999, 0), (1, 0)])
#     cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
#     lines, _ = watershed_workflow.utils.cut(line, cut)
#     assert (len(lines) == 2)
#     l1 = lines[0]
#     l2 = lines[1]
#     print(l1.coords[:])
#     print(l2.coords[:])
#     assert (l1 == shapely.geometry.LineString([(0, 0), (0.5, 0)]))
#     assert (l2 == shapely.geometry.LineString([(0.5, 0), (1, 0)]))


def test_cut_first_point():
    line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0)])
    cut = shapely.geometry.LineString([(0, -1), (0, 1)])
    lines, _ = watershed_workflow.utils.cut(line, cut)
    assert (len(lines) == 1)
    print(list(lines[0].coords))
    assert (watershed_workflow.utils.isClose(lines[0], line))



#
# NOTE: cut no longer snaps!
#

# def test_cut_nearly_first_point():
#     line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0)])
#     cut = shapely.geometry.LineString([(0.001, -1), (0.001, 1)])
#     lines, _ = watershed_workflow.utils.cut(line, cut)
#     assert (len(lines) == 1)
#     print(list(lines[0].coords))
#     assert (watershed_workflow.utils.isClose(
#         lines[0], shapely.geometry.LineString([(0.001, 0), (0.5, 0), (1, 0)])))


def test_cut_last_point():
    line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0)])
    cut = shapely.geometry.LineString([(1, -1), (1, 1)])
    lines, _ = watershed_workflow.utils.cut(line, cut)
    assert (len(lines) == 1)
    print(list(lines[0].coords))
    assert (watershed_workflow.utils.isClose(lines[0], line))


# def test_cut_nearly_last_point():
#     line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0)])
#     cut = shapely.geometry.LineString([(0.9999, -1), (0.9999, 1)])
#     lines, _ = watershed_workflow.utils.cut(line, cut)
#     assert (len(lines) == 1)
#     print(list(lines[0].coords))
#     assert (watershed_workflow.utils.isClose(
#         lines[0], shapely.geometry.LineString([(0., 0), (0.5, 0), (0.9999, 0)])))


def test_cut_two_crossings():
    line = shapely.geometry.LineString([(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1), (1.5, 1), (1.5, -1)])
    lines, _ = watershed_workflow.utils.cut(line, cut)
    assert (len(lines) == 3)
    print(list(lines[0].coords))
    assert (lines[0] == shapely.geometry.LineString([(0, 0), (0.5, 0)]))
    assert (lines[1] == shapely.geometry.LineString([(0.5, 0), (1, 0), (1.5, 0)]))
    assert (lines[2] == shapely.geometry.LineString([(1.5, 0), (2, 0)]))


def test_cut_two_ways():
    line1 = shapely.geometry.LineString([(-1, 0), (1, 0)])
    line2 = shapely.geometry.LineString([(0, -1), (0, 1)])
    l1_segs, l2_segs = watershed_workflow.utils.cut(line1, line2)
    assert (l1_segs[0] == shapely.geometry.LineString([(-1, 0), (0, 0)]))
    assert (l1_segs[1] == shapely.geometry.LineString([(0, 0), (1, 0)]))
    assert (l2_segs[0] == shapely.geometry.LineString([(0, -1), (0, 0)]))
    assert (l2_segs[1] == shapely.geometry.LineString([(0, 0), (0, 1)]))


def test_raises():
    line = shapely.geometry.LineString([(0, 0), (1, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.6, -1)])
    l1, c1 = watershed_workflow.utils.cut(line, cut)
    assert watershed_workflow.utils.isClose(l1[0], line)
    assert watershed_workflow.utils.isClose(c1[0], cut)


def test_intersect_point_to_linestring():
    from shapely.geometry import Point as P

    # test on the first point
    p0 = watershed_workflow.utils.intersectPointToSegment(P(0, 0), P(0, 0), P(0, 1))
    assert (watershed_workflow.utils.isClose(p0, (0, 0)))

    # test on the last point
    p0 = watershed_workflow.utils.intersectPointToSegment(P(0, 1), P(0, 0), P(0, 1))
    assert (watershed_workflow.utils.isClose(p0, (0, 1)))

    # test on the line
    p0 = watershed_workflow.utils.intersectPointToSegment(P(0, 0), P(0, -1), P(0, 1))
    assert (watershed_workflow.utils.isClose(p0, (0, 0)))

    # test x-perp
    p0 = watershed_workflow.utils.intersectPointToSegment(P(1, 0), P(0, -1), P(0, 1))
    assert (watershed_workflow.utils.isClose(p0, (0, 0)))

    # test diagonal perp
    p0 = watershed_workflow.utils.intersectPointToSegment(P(1, -1), P(-1, -1), P(1, 1))
    assert (watershed_workflow.utils.isClose(p0, (0, 0)))

    # test colinear but negative
    p0 = watershed_workflow.utils.intersectPointToSegment(P(-2, -2), P(-1, -1), P(1, 1))
    assert (watershed_workflow.utils.isClose(p0, (-1, -1)))

    # test colinear but positive
    p0 = watershed_workflow.utils.intersectPointToSegment(P(2, 2), P(-1, -1), P(1, 1))
    assert (watershed_workflow.utils.isClose(p0, (1, 1)))

    # test not colinear but past end
    p0 = watershed_workflow.utils.intersectPointToSegment(P(-3.3, -2.1), P(-1, -1), P(1, 1))
    assert (watershed_workflow.utils.isClose(p0, (-1, -1)))

    # test end but close
    p0 = watershed_workflow.utils.intersectPointToSegment(P(-.9, -1.1), P(-1, -1), P(1, 1))
    assert (watershed_workflow.utils.isClose(p0, (-1, -1)))

    # test throws
    with pytest.raises(AssertionError):
        p0 = watershed_workflow.utils.intersectPointToSegment(P(-.9, -1.1), P(1, 1), P(1, 1))


def test_neighborhood():
    p1 = shapely.geometry.LineString([(0, 0), (1, 1)])

    p2 = shapely.geometry.LineString([(2, 2), (3, 3)])
    assert (not watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(0, 3), (3, 3)])
    assert (not watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(3, 0), (3, 1)])
    assert (not watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(1, 0), (0, 1)])
    assert (watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(1, 1), (2, 1)])
    assert (watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(1.01, 1), (2, 1)])
    assert (watershed_workflow.utils.inNeighborhood(p1, p2))

    p2 = shapely.geometry.LineString([(1.01, 1), (2, 1)])
    assert (not watershed_workflow.utils.inNeighborhood(p1, p2, 1.e-3))

    # single point
    p3 = shapely.geometry.Point(978563.4249385255, 1512322.6640905372)
    p4 = shapely.geometry.LineString([(977132.6302807415, 1507051.5674243502),
                                      (979578.2010028946, 1515834.394320889)])
    assert (watershed_workflow.utils.inNeighborhood(p3, p4, 0.))


def test_generate_points():
    # with points
    p = shapely.geometry.Point(978563., 1512322.)
    with pytest.raises(TypeError):
        for r in watershed_workflow.utils.generateRings(shapely.geometry.mapping(p)):
            pass
    c1 = list(watershed_workflow.utils.generateCoords(shapely.geometry.mapping(p)))
    assert (1 == len(c1))
    assert (np.allclose(np.array([978563., 1512322.]), np.array(c1)))


def test_generate_lines():
    l1 = shapely.geometry.LineString([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])
    l1a = np.array([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])
    for r in watershed_workflow.utils.generateRings(shapely.geometry.mapping(l1)):
        assert (np.allclose(l1a, np.array(r)))

    for c1, c2 in zip(l1a, watershed_workflow.utils.generateCoords(shapely.geometry.mapping(l1))):
        assert (watershed_workflow.utils.isClose(tuple(c1), c2))


def test_generate_multilines():
    l1 = shapely.geometry.LineString([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])
    l1a = np.array([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])

    l2 = shapely.geometry.LineString([(2.5, 2.5), (3, 3), (4, 4), (5, 5)])
    l2a = np.array([(2.5, 2.5), (3, 3), (4, 4), (5, 5)])

    ml = shapely.geometry.MultiLineString([l1, l2])
    ringlist = list(watershed_workflow.utils.generateRings(shapely.geometry.mapping(ml)))
    assert (np.allclose(l1a, np.array(ringlist[0])))
    assert (np.allclose(l2a, np.array(ringlist[1])))

    coordlist = np.array(
        list(watershed_workflow.utils.generateCoords(shapely.geometry.mapping(ml))))
    assert (np.allclose(np.concatenate([l1a, l2a]), coordlist))


def test_generate_polygons():
    poly1 = shapely.geometry.Polygon([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])
    poly1a = np.array([(0, 0), (1, 1), (2, 2), (2.5, 2.5), (0, 0)])

    for r in watershed_workflow.utils.generateRings(shapely.geometry.mapping(poly1)):
        assert (np.allclose(poly1a, np.array(r)))

    for c1, c2 in zip(poly1a,
                      watershed_workflow.utils.generateCoords(shapely.geometry.mapping(poly1))):
        assert (watershed_workflow.utils.isClose(tuple(c1), c2))


def test_generate_multipolygons():
    poly1 = shapely.geometry.Polygon([(0, 0), (1, 1), (2, 2), (2.5, 2.5)])
    poly1a = np.array([(0, 0), (1, 1), (2, 2), (2.5, 2.5), (0, 0)])

    poly2 = shapely.geometry.Polygon([(2.5, 2.5), (3, 3), (4, 4), (5, 5)])
    poly2a = np.array([(2.5, 2.5), (3, 3), (4, 4), (5, 5), (2.5, 2.5)])

    mpoly = shapely.geometry.MultiPolygon([poly1, poly2])

    ringlist = list(watershed_workflow.utils.generateRings(shapely.geometry.mapping(mpoly)))
    assert (np.allclose(poly1a, np.array(ringlist[0])))
    assert (np.allclose(poly2a, np.array(ringlist[1])))

    coordlist = np.array(
        list(watershed_workflow.utils.generateCoords(shapely.geometry.mapping(mpoly))))
    assert (np.allclose(np.concatenate([poly1a, poly2a]), coordlist))


def test_angle():
    v1 = shapely.geometry.LineString([(0,0), (-1,0)])

    v2 = shapely.geometry.LineString([(0,1), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v2) - 90) < 1.e-10

    v3 = shapely.geometry.LineString([(1,1), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v3) - .75 * 180) < 1.e-10
    
    v4 = shapely.geometry.LineString([(1,0), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v4) - 180) < 1.e-10

    v5 = shapely.geometry.LineString([(1,-1), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v5) - 5. * 180 / 4.) < 1.e-10

    v6 = shapely.geometry.LineString([(0,-1), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v6) - 6 * 180 / 4.) < 1.e-10


    v7 = shapely.geometry.LineString([(-1,1.e-10), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v7) - 0.) < 1.e-8

    v8 = shapely.geometry.LineString([(-1,-1.e-10), (0,0)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v8) - 360) < 1.e-8


def test_project():
    v1 = shapely.geometry.LineString([(0,0), (-1,0)])

    v2_gold = shapely.geometry.LineString([(0,1), (0,0)])
    vec2 = watershed_workflow.utils.projectVectorAtAngle(v1,90,1)
    v2 = shapely.geometry.LineString([np.array(v1.coords[0]) + np.array(vec2), v1.coords[0]])
    assert watershed_workflow.utils.isClose(v2_gold, v2)

    v3_gold = shapely.geometry.LineString([(1,1), (0,0)])
    vec3 = watershed_workflow.utils.projectVectorAtAngle(v1, .75*180, np.sqrt(2))
    v3 = shapely.geometry.LineString([np.array(v1.coords[0]) + np.array(vec3), v1.coords[0]])
    assert watershed_workflow.utils.isClose(v3_gold, v3)
    

    vec4 = watershed_workflow.utils.projectVectorAtAngle(v1, .75*180, 4)
    assert np.allclose(vec4, np.array(vec3) * 4. / np.linalg.norm(vec3), 1.e-10)

    
def test_angle2():
    v1 = shapely.geometry.LineString([(1,1), (1,2)])

    v2 = shapely.geometry.LineString([(2,1), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v2) - 90) < 1.e-10

    v3 = shapely.geometry.LineString([(2,0), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v3) - .75 * 180) < 1.e-10
    
    v4 = shapely.geometry.LineString([(1,0), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v4) - 180) < 1.e-10

    v5 = shapely.geometry.LineString([(0,0), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v5) - 5. * 180 / 4.) < 1.e-10

    v6 = shapely.geometry.LineString([(0,1), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v6) - 6 * 180 / 4.) < 1.e-10


    v7 = shapely.geometry.LineString([(1+1.e-10,2), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v7) - 0.) < 1.e-8

    v8 = shapely.geometry.LineString([(1-1.e-10,2), (1,1)])
    assert abs(watershed_workflow.utils.computeAngle(v1,v8) - 360) < 1.e-8


def test_project2():
    v1 = shapely.geometry.LineString([(1,1), (1,2)])

    v2_gold = shapely.geometry.LineString([(2,1), (1,1)])
    vec2 = watershed_workflow.utils.projectVectorAtAngle(v1,90,1)
    v2 = shapely.geometry.LineString([np.array(v1.coords[0]) + np.array(vec2), v1.coords[0]])
    assert watershed_workflow.utils.isClose(v2_gold, v2)

    v3_gold = shapely.geometry.LineString([(2,0), (1,1)])
    vec3 = watershed_workflow.utils.projectVectorAtAngle(v1, .75*180, np.sqrt(2))
    v3 = shapely.geometry.LineString([np.array(v1.coords[0]) + np.array(vec3), v1.coords[0]])
    assert watershed_workflow.utils.isClose(v3_gold, v3)
    

    vec4 = watershed_workflow.utils.projectVectorAtAngle(v1, .75*180, 4)
    assert np.allclose(vec4, np.array(vec3) * 4. / np.linalg.norm(vec3), 1.e-10)

    
    


    
    
