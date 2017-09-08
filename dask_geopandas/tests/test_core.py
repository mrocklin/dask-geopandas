import dask_geopandas as dg
import dask.dataframe as dd
import geopandas as gpd
import random

import pytest
from shapely.geometry import Polygon, Point
import dask
import numpy as np
import pandas as pd

triangles = [Polygon([(0, 0), (1, 0), (0, 1)]),
             Polygon([(1, 0), (1, 1), (0, 1)])]
tri_df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})

fine_delta = 0.1
inds = np.arange(0, 1, fine_delta)
grid = [Polygon([(x, y), (x + fine_delta, y),
                 (x + fine_delta, y + fine_delta), (x, y + fine_delta)])
        for x in inds
        for y in inds]
grid_df = gpd.GeoDataFrame({'x': np.concatenate([inds] * 10),
                            'y': np.repeat(inds, 10),
                            'geometry': grid})

coarse_delta = 0.3
coarse_inds = np.arange(0, 1, coarse_delta)
coarse_grid = [Polygon([(x, y), (x + coarse_delta, y),
                       (x + coarse_delta, y + coarse_delta), (x, y + coarse_delta)])
               for x in coarse_inds
               for y in coarse_inds]

points = [Point(random.random(), random.random()) for i in range(200)]
points_df = gpd.GeoDataFrame({'value': np.random.random(len(points)),
                              'geometry': points})


def assert_eq(a, b):
    if hasattr(a, 'columns'):
        assert list(a.columns) == list(b.columns)
    if hasattr(a, 'dtype'):
        assert a.dtype == b.dtype
    if hasattr(a, 'dtypes'):
        assert (a.dtypes == b.dtypes).all()

    aa = a
    bb = b
    if hasattr(a, 'dask'):
        aa = a.compute(get=dask.get)
    if hasattr(b, 'dask'):
        bb = b.compute(get=dask.get)

    assert str(aa) == str(bb)


@pytest.mark.parametrize('npartitions', [1, 2])
def test_basic(npartitions):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)
    assert ddf.npartitions == npartitions
    assert_eq(ddf, df)
    assert_eq(ddf.x, df.x)
    assert_eq(ddf.x + 1, df.x + 1)

@pytest.mark.parametrize('npartitions', [1, 2])
@pytest.mark.parametrize('op', [
    'contains',
    'geom_equals',
    'geom_almost_equals',
    'crosses',
    'disjoint',
    'intersects',
    'overlaps',
    'touches',
    'within',
    'distance',
    'difference',
    'symmetric_difference',
    'union',
    'intersection',
    'distance',
])
def test_binary_singleton(npartitions, op):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)

    def func(a, b):
        return getattr(a, op)(b)

    point = Point(0.2, 0.2)
    result = func(ddf, point)
    assert_eq(func(ddf, point), func(df, point))
    assert_eq(func(ddf.geometry, point), func(df.geometry, point))


@pytest.mark.parametrize('npartitions', [1, 2])
@pytest.mark.parametrize('op', [
    'area',
    'length',
])
def test_unary_property(npartitions, op):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)

    def func(a):
        return getattr(a, op)

    point = Point(0.2, 0.2)
    assert_eq(func(ddf), func(df))
    assert_eq(func(ddf.geometry), func(df.geometry))


@pytest.mark.parametrize('npartitions', [1, 2])
@pytest.mark.parametrize('op,args', [
    ('buffer', (1,)),
])
def test_unary_method(npartitions, op, args):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)

    def func(a):
        return getattr(a, op)(*args)

    point = Point(0.2, 0.2)
    assert_eq(func(ddf), func(df))
    assert_eq(func(ddf.geometry), func(df.geometry))


def test_head():
    ddf = dg.from_pandas(points_df, npartitions=5)
    head = ddf.head(5)
    assert isinstance(head, gpd.GeoDataFrame)
    assert_eq(head, points_df.head(5))


def test_persist():
    df = dg.from_pandas(points_df, npartitions=5)
    df = dg.repartition(df, triangles)

    df2 = df.persist()
    assert not any(map(dask.core.istask, df2.dask.values()))


@pytest.mark.parametrize('func', [str, repr])
def test_repr(func):
    ddf = dg.from_pandas(points_df, npartitions=5)
    for x in [ddf, ddf.value, ddf.geometry]:
        text = func(x)
        assert '5' in text
        assert type(x).__name__ in text


def test_repartition():
    df = points_df
    df = dg.repartition(df, triangles).persist()
    assert df.npartitions == 2
    assert len(df) == len(points_df)
    assert df._regions.iloc[0].equals(triangles[0])
    assert df._regions.iloc[1].equals(triangles[1])

    for i in range(df.npartitions):
        part = df.get_partition(i)
        geoms = part.geometry.compute()
        assert geoms.within(part._regions.iloc[0]).all()

    df2 = df.repartition(grid).persist()
    assert len(df2) == len(points_df)
    assert df2.npartitions == len(grid)

    for i in range(df2.npartitions):
        part = df2.get_partition(i)
        geoms = part.geometry.compute()
        assert geoms.within(part._regions.iloc[0]).all()


def test_repartition_polys():
    with dask.set_options(get=dask.get):
        df = dg.from_pandas(grid_df, npartitions=3)
        assert len(df) == len(grid_df)
        df = dg.repartition(df, triangles)
        # assert len(df) == len(grid_df)  # fails because touches fails
        df = df.persist()

        assert df.npartitions == 2

        for i in range(df.npartitions):
            part = df.get_partition(i)
            geoms = part.geometry.compute()
            assert geoms.within(part._regions.iloc[0]).all()


def test_repartition_pandas_expands_regions():
    df = dg.repartition(grid_df, triangles)

    for i in range(df.npartitions):
        part = df.get_partition(i)
        geoms = part.geometry.compute()
        assert geoms.within(part._regions.iloc[0]).all()

    assert len(df) == len(grid_df)


def test_repartition_trim():
    polys = triangles + [Polygon([(10, 10), (10, 20), (20, 10)])]
    df = dg.repartition(grid_df, polys)

    assert df.npartitions == 2


def test_repartition_single_element():
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': [Point(0, 0), Point(1, 1)]})
    gdf = dg.repartition(df, triangles)
    assert len(df) == 2


def test_len():
    df = points_df
    ddf = dg.from_pandas(df, npartitions=3)
    assert len(ddf) == len(df)


def test_sjoin():
    import shapely.affinity
    l = dg.repartition(points_df, [shapely.affinity.translate(t, 0.00001, 0) for t
        in triangles])
    r = dg.repartition(grid_df, coarse_grid)

    result = dg.sjoin(l, r, how='inner', op='intersects')
    expected = gpd.sjoin(points_df, grid_df, how='inner', op='intersects')
    assert len(result) == len(expected)
    assert (result._regions.area < (coarse_delta + 0.1) **2 ).all()


def test_set_geometry():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                       'y': [2, 3, 4, 5, 6],
                       'value': [1, 1, 1, 2, 2]})
    ddf = dd.from_pandas(df, npartitions=2)
    gdf = ddf.set_geometry(ddf[['x', 'y']])
    assert isinstance(gdf, dg.GeoDataFrame)
    assert gdf.npartitions == ddf.npartitions

    expected = gpd.vectorized.points_from_xy(df.x.values, df.y.values)
    assert gdf.geometry.compute().equals(gpd.GeoSeries(expected))
    assert isinstance(gdf.compute(), gpd.GeoDataFrame)
    for c in ['x', 'y', 'value']:
        assert c in gdf.columns
