Dask Geopandas
==============

Parallel GeoPandas with Dask

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

We can also repartition by a set of known regions.  This suffers an an upfront
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
