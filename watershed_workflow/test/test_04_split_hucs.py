import pytest
import shapely.geometry
import geopandas
import watershed_workflow.split_hucs

from watershed_workflow.test.shapes import two_boxes, three_boxes


def test_hc():
    """HandledCollection is in the inner guts of split_hucs"""
    # construction
    things = ['a', 'b', 'c', 'd']
    hc = watershed_workflow.split_hucs.HandledCollection(things)
    assert (len(hc) == 4)
    for i, j in zip(things, hc):
        assert i == j

    # addition
    hc.append('e')
    assert (len(hc) == 5)

    # removal
    hc.pop(3)

    # iteration
    remaining = ['a', 'b', 'c', 'e']
    for i, j in zip(remaining, hc):
        assert i == j


def test_intersect_and_split(two_boxes):
    boundaries, intersections = watershed_workflow.split_hucs.intersectAndSplit(two_boxes.geometry)
    assert (len(boundaries) is 2)

    for b in boundaries:
        assert type(b) is shapely.geometry.LineString
        assert len(b.coords) == 4

    assert (len(intersections) is 2)
    for i, row in enumerate(intersections):
        assert (len(row) is 2)
        for j, entry in enumerate(row):
            print("At i,j=%d,%d type is %r" % (i, j, type(entry)))
            if i <= j:
                assert entry is None
            else:
                assert type(entry) is shapely.geometry.LineString
                assert len(entry.coords) is 2
                watershed_workflow.utils.isClose(entry.coords[0], (10, -5))
                watershed_workflow.utils.isClose(entry.coords[1], (10, 5))


def test_hucs(two_boxes):
    # test construction
    tb = watershed_workflow.split_hucs.SplitHUCs(two_boxes)

    # test uniqueness of the boundaries+intersections collections
    handles = [p for b in tb.boundaries for p in b] + [p for i in tb.intersections for p in i]
    assert (len(handles) == 3)
    assert (len(set(handles)) == 3)
    assert (handles == [0, 1, 2])

    # test correctness of keys
    assert (len(tb.boundaries) == 2)
    assert (list(tb.boundaries.handles()) == [0, 1])

    assert (len(tb.intersections) == 1)
    assert (list(tb.intersections.handles()) == [0, ])

    assert (len(tb.linestrings) == 3)
    assert (list(tb.linestrings.handles()) == [0, 1, 2])

    # gons
    b, i = tb.gons[0]
    assert (list(b) == [0, ])
    assert (list(i) == [0, ])

    b, i = tb.gons[1]
    assert (list(b) == [1, ])
    assert (list(i) == [0, ])

    p0 = tb.computePolygon(0)
    assert (len(p0.boundary.coords) == 5)
    assert (watershed_workflow.utils.isClose(two_boxes.geometry[0], p0))

    p1 = tb.computePolygon(1)
    assert (len(p1.boundary.coords) == 5)
    assert (watershed_workflow.utils.isClose(two_boxes.geometry[1], p1))

    # boundary gon
    p3 = tb.exterior
    assert (len(p3.boundary.coords) == 7)  # closed polygon
    bndry_c = [(0, -5), (10, -5), (20, -5), (20, 5), (10, 5), (0, 5)]
    poly = shapely.geometry.Polygon(bndry_c)
    assert (watershed_workflow.utils.isClose(poly, p3))

    # should check that these are close to those in two_boxes, but
    # they are shifted, so this check would be difficult.
    # for b1,b2 in zip(tb.polygons(), two_boxes):
    #     assert(watershed_workflow.utils.isClose(b1,b2))

    # now split the middle
    # one could imagine iterating over the spine and smoothing/doing something
    for spine in tb.intersections:
        assert (len(spine) is 1)
        int_handle, seg_handle = next(spine.items())
        seg = tb.linestrings[seg_handle]
        # split seg into two
        assert (len(seg.coords) == 2)
        seg1 = shapely.geometry.LineString([seg.coords[0], (10., 0.)])
        seg2 = shapely.geometry.LineString([(10., 0.), seg.coords[1]])
        tb.linestrings.pop(seg_handle)
        new_seg_handles = tb.linestrings.extend([seg1, seg2])
        spine.pop(int_handle)
        spine.extend(new_seg_handles)

    # now check that the polygons have 5 coordinates (+1 for repeated start/end)
    p0 = tb.computePolygon(0)
    p1 = tb.computePolygon(1)
    assert (len(p0.boundary.coords) == 6)
    assert (len(p1.boundary.coords) == 6)
    print(list(p0.boundary.coords))
    print(list(p1.boundary.coords))


def test_hucs_three(three_boxes):
    # test construction
    tb = watershed_workflow.split_hucs.SplitHUCs(three_boxes)

    # test uniqueness of the boundaries+intersections collections
    handles = [p for b in tb.boundaries for p in b] + [p for i in tb.intersections for p in i]
    assert (len(handles) == 6)
    assert (len(set(handles)) == 6)

    # test correctness of keys
    assert (len(tb.boundaries) == 4)
    assert (len(tb.intersections) == 2)


def test_hucs_triple():
    b1 = [(0, -5), (10, -5), (10., 5.), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10., 5.)]
    b3 = [(0, 5), (10., 5.), (20, 5), (20, 10), (0, 10)]
    boxes = [
        shapely.geometry.Polygon(b1),
        shapely.geometry.Polygon(b2),
        shapely.geometry.Polygon(b3),
    ]
    df = geopandas.GeoDataFrame({'index' : range(len(boxes)),
                                 'geometry' : boxes})
    hucs = watershed_workflow.split_hucs.SplitHUCs(df)
    assert (len(hucs) is 3)
    assert (len(hucs.linestrings) is 6)
    assert (len(hucs.intersections) is 3)
    assert (len(hucs.boundaries) is 3)

    # note order is not required here, but I don't have a good way of checking without order
    boundaries = [shandle for b in hucs.boundaries for shandle in b]
    bound1 = hucs.linestrings[boundaries[0]]
    assert (watershed_workflow.utils.isClose(bound1,
                                           shapely.geometry.LineString([(0, 5), (0, -5),
                                                                        (10, -5)])))

    bound2 = hucs.linestrings[boundaries[1]]
    assert (watershed_workflow.utils.isClose(
        bound2, shapely.geometry.LineString([(10, -5), (20, -5), (20, 5)])))

    bound3 = hucs.linestrings[boundaries[2]]
    assert (watershed_workflow.utils.isClose(
        bound3, shapely.geometry.LineString([(20, 5), (20, 10), (0, 10), (0, 5)])))

    intersections = [shandle for b in hucs.intersections for shandle in b]
    spine1 = hucs.linestrings[intersections[0]]
    assert (watershed_workflow.utils.isClose(spine1, shapely.geometry.LineString([(10, 5),
                                                                                (10, -5)])))

    spine2 = hucs.linestrings[intersections[1]]
    assert (watershed_workflow.utils.isClose(spine2, shapely.geometry.LineString([(0, 5), (10, 5)])))

    spine3 = hucs.linestrings[intersections[2]]
    assert (watershed_workflow.utils.isClose(spine3, shapely.geometry.LineString([(10, 5), (20, 5)])))
