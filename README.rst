Dask Geopandas
==============

Parallel GeoPandas with Dask

Status
------

*This project is not in a functional state and should not be relied upon.
No guarantee of support is provided.*

This was was originally implemented to demonstrate speedups from parallelism
alongside an experimental Cythonized branch of GeoPandas.  That cythonized
branch has since evolved to the point where the code here no longer works with
the latest version.

If you *really* want to get this to work then you should checkout the
geopandas-cython branch of geopandas at about 2017-09-21 and build from source
(this may not be fun).  But really the solution is probably to wait until
everything settles.  There is no known timeline for this.

If you would like to see this project in a more stable state then you might
consider pitching in with developer time or with financial support from you or
your company.


Example
-------

Given a GeoPandas dataframe

.. code-block:: python

   import geopandas as gpd
   df = gpd.read_file('...')

We can repartition it into a Dask-GeoPandas dataframe either naively by rows.
This does not provide a spatial partitioning and so won't gain the efficiencies
of spatial reasoning, but will still provide basic multi-core parallelism.

.. code-block:: python

   import dask_geopandas as dg
   ddf = dg.from_pandas(df, npartitions=4)

We can also repartition by a set of known regions.  This suffers an upfront
cost of a spatial join, but enables spatial-aware computations in the future to
be faster.

.. code-block:: python

   regions = gpd.read_file('boundaries.shp')
   ddf = dg.repartition(df, regions)

Additionally, if you have a distributed dask.dataframe you can pass columns of
x-y points to the `set_geometry` method.  Currently this only supports point
data.

.. code-block:: python

   import dask.dataframe as dd
   import dask_geopandas as dg

   df = dd.read_csv('...')

   df = df.set_geometry(df[['latitude', 'longitude']])
