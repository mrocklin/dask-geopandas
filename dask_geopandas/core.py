import random
import sys

import geopandas as gpd
import pandas as pd
from dask.utils import M, funcname
from dask.base import tokenize, normalize_token
from dask.optimize import key_split
from dask.compatibility import apply
import dask.dataframe as dd
import dask.threaded
import dask.base
import shapely
import shapely.ops
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.geometry import Polygon, Point
from toolz import merge

import operator


def typeof(example):
    if isinstance(example, gpd.GeoDataFrame):
        return GeoDataFrame
    elif isinstance(example, gpd.GeoSeries):
        return GeoSeries
    elif isinstance(example, pd.Series):
        return Series
    else:
        raise TypeError()


class GeoFrame(dask.base.Base):
    _default_get = staticmethod(dask.threaded.get)

    @staticmethod
    def _finalize(results):
        if isinstance(results[0], (gpd.GeoSeries, gpd.GeoDataFrame)):
            return gpd.concat(results)
        else:
            return pd.concat(results)

    def __init__(self, dsk, name, regions, example):
        if not isinstance(regions, gpd.GeoSeries):
            regions = gpd.GeoSeries(regions)
        self._regions = regions
        self._example = example
        self.dask = dsk
        self._name = name

    def __str__(self):
        return "<%s: %s, npartitions=%d>" % (type(self).__name__,
                key_split(self._name), self.npartitions)

    __repr__ = __str__

    def __len__(self):
        return len(self.compute())  # TODO: map_partitions(len).sum().compute()

    def copy(self):
        return type(self)(self.dask, self._name, self._regions, self._example)

    def __getstate__(self):
        return self.dask, self._name, self._regions, self._example

    def __setstate__(self, state):
        self.dask, self._name, self._regions, self._example = state

    def plot(self):
        return self._regions.plot()

    def _keys(self):
        return [(self._name, i) for i in range(len(self._regions))]

    @property
    def npartitions(self):
        return len(self._regions)

    def map_partitions(self, func, *args, **kwargs):
        example = func(self._example, *args, **kwargs)
        name = funcname(func) + '-' + tokenize(self, func, *args, **kwargs)
        if not args and not kwargs:
            dsk = {(name, i): (func, key) for i, key in enumerate(self._keys())}
        else:
            dsk = {(name, i): (apply, func, list((key,) + args), kwargs)
                   for i, key in enumerate(self._keys())}
        return typeof(example)(merge(dsk, self.dask), name,
                               self._regions, example)

    def get_partition(self, n):
        name = 'get-partition-%d-%s' % (n, tokenize(self))
        dsk = {(name, 0): (self._name, n)}
        return type(self)(merge(dsk, self.dask), name,
                          self._regions.iloc[n:n + 1], self._example)

    def head(self, n=5, compute=True):
        result = self.get_partition(0).map_partitions(M.head, n)
        if compute:
            result = result.compute()
        return result

    def __add__(self, other):
        return self.map_partitions(operator.add, other)

    def __mul__(self, other):
        return self.map_partitions(operator.mul, other)

    def __mod__(self, other):
        return self.map_partitions(operator.mod, other)

    def __truediv__(self, other):
        return self.map_partitions(operator.truediv, other)

    def __floordiv__(self, other):
        return self.map_partitions(operator.floordiv, other)

    @property
    def area(self):
        return self.map_partitions(lambda x: x.area)

    @property
    def geom_type(self):
        return self.map_partitions(lambda x: x.geom_type)

    @property
    def type(self):
        return self.map_partitions(lambda x: x.type)

    @property
    def length(self):
        return self.map_partitions(lambda x: x.length)

    @property
    def is_valid(self):
        return self.map_partitions(lambda x: x.is_valid)

    @property
    def is_empty(self):
        return self.map_partitions(lambda x: x.is_empty)

    @property
    def is_simple(self):
        return self.map_partitions(lambda x: x.is_simple)

    @property
    def is_ring(self):
        return self.map_partitions(lambda x: x.is_ring)

    @property
    def boundary(self):
        return self.map_partitions(lambda x: x.boundary)

    @property
    def centroid(self):
        return self.map_partitions(lambda x: x.centroid)

    @property
    def convex_hull(self):
        return self.map_partitions(lambda x: x.convex_hull)

    @property
    def exterior(self):
        return self.map_partitions(lambda x: x.exterior)

    def representative_point(self):
        return self.map_partitions(M.representative_point)

    def contains(self, other):
        return self.map_partitions(M.contains, other)

    def geom_equals(self, other):
        return self.map_partitions(M.geom_equals, other)

    def geom_almost_equals(self, other, decimal=6):
        return self.map_partitions(M.geom_almost_equals, other, decimal=6)

    def crosses(self, other):
        return self.map_partitions(M.crosses, other)

    def disjoint(self, other):
        return self.map_partitions(M.disjoint, other)

    def intersects(self, other):
        return self.map_partitions(M.intersects, other)

    def overlaps(self, other):
        return self.map_partitions(M.overlaps, other)

    def touches(self, other):
        return self.map_partitions(M.touches, other)

    def within(self, other):
        return self.map_partitions(M.within, other)

    def distance(self, other):
        return self.map_partitions(M.distance, other)

    def difference(self, other):
        return self.map_partitions(M.difference, other)

    def symmetric_difference(self, other):
        return self.map_partitions(M.symmetric_difference, other)

    def union(self, other):
        return self.map_partitions(M.union, other)

    def intersection(self, other):
        return self.map_partitions(M.intersection, other)

    def buffer(self, distance, resolution=16, cap_style=CAP_STYLE.round,
               join_style=JOIN_STYLE.round, mitre_limit=5.0):
        df = self.map_partitions(M.buffer, distance, resolution=resolution,
                cap_style=cap_style, join_style=join_style,
                mitre_limit=mitre_limit)
        df._regions = df._regions.buffer(distance, resolution=resolution,
                cap_style=cap_style, join_style=join_style,
                mitre_limit=mitre_limit)
        return df

    def repartition(self, partitions, trim=True, duplicate=False):
        return repartition(self, partitions, trim=trim, duplicate=duplicate)


class GeoDataFrame(GeoFrame):
    def __getitem__(self, key):
        if isinstance(key, str) and key in self.columns:
            return self.map_partitions(operator.getitem, key)
        raise NotImplementedError()

    def __getattr__(self, key):
        if key in self.columns:
            return self.map_partitions(getattr, key)
        else:
            raise AttributeError("GeoDataFrame has no attribute %r" % key)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(c for c in self.columns if
                 (isinstance(c, pd.compat.string_types) and
                 pd.compat.isidentifier(c)))
        return list(o)

    @property
    def columns(self):
        return self._example.columns

    @property
    def dtype(self):
        return self._example.dtype


class GeoSeries(GeoFrame):
    @property
    def name(self):
        return self._example.name

    @property
    def crs(self):
        return self._example.crs


class Series(GeoFrame):
    @property
    def dtype(self):
        return self._example.dtype

    @property
    def name(self):
        return self._example.name


inf = sys.float_info.max / 10
all_space = shapely.geometry.Polygon([(inf, inf), (inf, -inf),
                                      (-inf, -inf), (-inf, inf)])

def from_pandas(df, npartitions=4):

    blocksize = len(df) // npartitions
    name = 'from-pandas-' + tokenize(df, npartitions)
    if npartitions > 1:
        dsk = {(name, i): df.iloc[i * blocksize: (i + 1) * blocksize]
               for i in range(0, len(df) // blocksize)}
        i = npartitions - 1
        dsk[name, i] = df.iloc[blocksize * i:]
    else:
        dsk = {(name, 0): df}

    return GeoDataFrame(dsk, name, [all_space] * npartitions, df.head(0))


@normalize_token.register(GeoFrame)
def _normalize_geoframe(gdf):
    return gdf._name


def _repartition_pandas(df, partitions):
    partitions = gpd.GeoDataFrame({'geometry': partitions.geometry},
                                  crs=partitions.crs)
    joined = gpd.sjoin(partitions, df, how='inner', op='intersects').index_right
    name = 'from-geopandas-' + tokenize(df, partitions)
    dsk = {}
    new_partitions = []

    j = 0
    for i, partition in enumerate(partitions.geometry):
        try:
            ind = joined.loc[i]
        except KeyError:
            continue
        else:
            if not isinstance(ind, pd.Series):
                ind = pd.Series([ind])
            subset = df.loc[ind]
        subset2 = subset[subset.geometry.representative_point().intersects(partition)]
        dsk[name, j] = subset2
        j += 1
        if (subset2.geometry.type == 'Point').all():
            new_partitions.append(partition)
        else:
            new_partitions.append(subset2.geometry.unary_union)

    result = GeoDataFrame(dsk, name, new_partitions, df.head(0))
    return result


def repartition(df, partitions, trim=True, duplicate=False):
    """ Repartition a GeoDataFrame along new partitions

    Parameters
    ----------
    df: GeoDataFrame
    partitions: collection of Geometries
    trim: boolean
        Whether or not to include partitions that may be empty
    duplicate: boolean (default to False)
        Whether or not to include geometries in multiple partitions if they
        intersect.  Otherwise only include a geometry if its representative
        point intersects with the region
    """
    if isinstance(partitions, GeoDataFrame):
        partitions = partitions.geometry
    elif not isinstance(partitions, gpd.GeoSeries):
        partitions = gpd.GeoSeries(partitions)
    if isinstance(df, (gpd.GeoSeries, gpd.GeoDataFrame)):
        return _repartition_pandas(df, partitions)

    partitions = gpd.GeoDataFrame({'geometry': partitions}, crs=partitions.crs)
    df_geom = gpd.GeoDataFrame({'geometry': df._regions})

    joined = gpd.sjoin(partitions, df_geom, how='inner')
    token = tokenize(df, partitions)
    name = 'repartition-pre-' + token
    dsk = {}
    regions = []

    for i, (j, (ind, geom)) in enumerate(joined.iterrows()):
        dsk[(name, i)] = (_subset_geom, (df._name, ind), geom, duplicate)
        geom2 = df._regions.loc[ind]
        regions.append(geom2)

    new_name = 'repartition-' + token
    ind = pd.Series(joined.index)
    groups = ind.groupby(ind)

    regions2 = []
    i = 0
    for group in groups.groups.values():
        if len(group) or not trim:
            dsk[(new_name, i)] = (gpd.concat, [(name, j) for j in group])
            region = shapely.ops.unary_union([regions[j] for j in group])
            regions2.append(region)
            i += 1

    return GeoDataFrame(merge(dsk, df.dask), new_name, regions2, df.head(0))


trans_x = 0.001# (random.random() - 0.5) / 1e100
trans_y = 0.002# (random.random() - 0.5) / 1e100


def _subset_geom(df, geom, duplicate=False):
    if duplicate:
        return df[df.geometry.intersects(geom)]
    rep = df.geometry.representative_point()
    intersects = rep.intersects(geom)
    touches = rep.touches(geom)
    if not touches.any():
        return df[intersects]
    else:
        translated = shapely.affinity.translate(geom, trans_x, trans_y)
        touched = rep[touches]
        intersects2 = touched.intersects(translated)
        return df[intersects ^ intersects2]


def sjoin(left, right, how='inner', op='intersects', buffer=0.01):
    name = 'sjoin-' + tokenize(left, right, how, op)
    example = gpd.tools.sjoin(left._example, right._example, how=how, op=op)

    parts = gpd.tools.sjoin(left._regions.to_frame(),
                            right._regions.to_frame(),
                            how='inner', op='intersects')
    left_parts = parts._geometry_array
    right_parts = right._regions._geometry_array[parts.index_right.values]

    dsk = {}
    regions = []
    for i, (l, (r, _)) in enumerate(parts.iterrows()):
        dsk[name, i] = (gpd.tools.sjoin, (left._name, l), (right._name, r), op, how)
        lr = left._regions.iloc[l]
        rr = right._regions.iloc[l]
        region = lr.intersection(rr).buffer(buffer).intersection(lr.union(rr))
        regions.append(region)

    return GeoDataFrame(merge(dsk, left.dask, right.dask), name, regions, example)


def _points_from_xy(x, y, crs=None):
    points = gpd.vectorized.points_from_xy(x.values, y.values)
    return gpd.GeoSeries(points, index=x.index, crs=crs)


def points_from_xy(x, y, crs=None):
    s = dd.map_partitions(_points_from_xy, x, y, crs=crs)
    example = gpd.GeoSeries(Point(0, 0))
    return GeoSeries(s.dask, s._name, [all_space] * s.npartitions, example)


def set_geometry(df, geometry, crs=None):
    if isinstance(geometry, dd.DataFrame) and len(geometry.columns) == 2:
        a, b = geometry.columns
        geometry = points_from_xy(geometry[a], geometry[b], crs=crs)

    assert df.npartitions == geometry.npartitions

    name = 'set-geometry-' + tokenize(df, geometry)
    dsk = {(name, i): (M.set_geometry, (df._name, i), (geometry._name, i),
        False, False, crs)
            for i in range(df.npartitions)}
    example = df._meta.set_geometry(geometry._example)

    gdf = GeoDataFrame(merge(df.dask, geometry.dask, dsk),
                       name, geometry._regions, example)
    return gdf


if sys.version_info[0] == 3:
    dd.DataFrame.set_geometry = set_geometry
else:
    import types
    dd.DataFrame.set_geometry = types.MethodType(set_geometry, None,
            dd.DataFrame)
