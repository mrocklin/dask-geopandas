import dask_geopandas as dg
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

delta = 0.1
inds = np.arange(0, 1, delta)
grid = [Polygon([(x, y), (x + delta, y),
                 (x + delta, y + delta), (x, y + delta)])
        for x in inds
        for y in inds]
grid_df = gpd.GeoDataFrame({'x': np.concatenate([inds] * 10),
                            'y': np.repeat(inds, 10),
                            'geometry': grid})

points = [Point(random.random(), random.random()) for i in range(100)]
points_df = gpd.GeoDataFrame({'value': np.random.random(100),
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
    'geom_equals',
    'distance',
    'contains',
    'intersects',
    'touches'
])
def test_binary_singleton(npartitions, op):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)

    def func(a, b):
        return getattr(a, op)(b)

    point = Point(0.2, 0.2)
    assert_eq(func(ddf, point), func(df, point))
    assert_eq(func(ddf.geometry, point), func(df.geometry, point))


@pytest.mark.parametrize('npartitions', [1, 2])
@pytest.mark.parametrize('op', [
    'area',
    'length',
])
def test_unary(npartitions, op):
    df = gpd.GeoDataFrame({'x': [1, 2], 'geometry': triangles})
    ddf = dg.from_pandas(df, npartitions=npartitions)

    def func(a):
        return getattr(a, op)

    point = Point(0.2, 0.2)
    assert_eq(func(ddf), func(df))
    assert_eq(func(ddf.geometry), func(df.geometry))


def test_partition():
    df = dg.partition(points_df, tri_df)
    assert df.npartitions == len(tri_df)
    assert set(df.value.compute()) == set(points_df.value)

    for i in range(df.npartitions):
        part = df.get_partition(i)
        geoms = part.geometry.compute()
        assert geoms.within(part._regions.iloc[0]).all()


def test_head():
    ddf = dg.from_pandas(points_df, npartitions=5)
    head = ddf.head(5)
    assert isinstance(head, gpd.GeoDataFrame)
    assert_eq(head, points_df.head(5))


@pytest.mark.parametrize('func', [str, repr])
def test_repr(func):
    ddf = dg.from_pandas(points_df, npartitions=5)
    text = func(ddf)
    assert '5' in text
    assert 'dataframe' in text.lower()
    assert 'from-pandas' in text.lower()
