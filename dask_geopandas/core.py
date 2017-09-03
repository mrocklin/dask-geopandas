import geopandas as gpd
import pandas as pd
from dask.utils import M, funcname
from dask.base import tokenize, normalize_token
from dask.optimize import key_split
from dask.compatibility import apply
import dask.threaded
import dask.base
import shapely
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

    def plot(self):
        return self._regions.plot()

    def _keys(self):
        return [(self._name, i) for i in range(len(self._regions))]

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

    def buffer(self, distance, resolution=16, cap_style=gpd.CAP_STYLE.round,
               join_style=gpd.JOIN_STYLE.round, mitre_limit=5.0):
        df = self.map_partitions(M.buffer, distance, resolution=resolution,
                cap_style=cap_style, join_style=join_style,
                mitre_limit=mitre_limit)
        df._regions = df._regions.buffer(distance, resolution=resolution,
                cap_style=cap_style, join_style=join_style,
                mitre_limit=mitre_limit)
        return df


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

    @property
    def npartitions(self):
        return len(self._regions)


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


inf = 1e30

def from_pandas(df, npartitions=4):
    region = shapely.geometry.Polygon([(inf, inf), (inf, -inf),
                                       (-inf, -inf), (-inf, inf)])

    blocksize = len(df) // npartitions
    name = 'from-pandas-' + tokenize(df, npartitions)
    dsk = {(name, i): df.iloc[i: i + blocksize]
           for i in range(0, len(df), blocksize)}
    i = npartitions - 1
    dsk[name, i] = df.iloc[blocksize * i:]

    return GeoDataFrame(dsk, name, [region] * npartitions, df.head(0))


@normalize_token.register(GeoFrame)
def _normalize_geoframe(gdf):
    return gdf._name


def partition(df, partitions):
    partitions = gpd.GeoDataFrame({'geometry': partitions.geometry},
                                  crs=partitions.crs)
    j = gpd.sjoin(partitions, df, how='inner').index_right
    name = 'from-pandas-' + tokenize(df, partitions)
    dsk = {(name, i): df.loc[j.loc[i]] for i in range(len(partitions))}

    return GeoDataFrame(dsk, name, partitions.geometry, df.head(0))
