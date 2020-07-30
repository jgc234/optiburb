# Optiburb Cycle Route Generator

This is a rather rough script to generate a GPX file for the optimum
"burbing" route.  Burbing is the process of riding every road in a
suburb or locality.

This script take the fun out of trying to work it out yourself.

## How it works.

Given a suburb name, Optiburb fetches data from OpenStreetMap and
tries to calculate the optimum route aroud the suburb, and spits out a
GPX file.

The solution is based on the Route Inspection Problem (AKA Chinese
Postman Problem), which is a well-known mathematical problem and
algorithm.  There is no amazing code in here - This program is calling
code from libraries to do all the heavy lifting for OSM, graph theory
and algorithms.  This program just glues them together.

There are some options to prune out unnamed gravel tracks, to import
boundaries from shapefiles

## Limitations

Some known limitations:

* The program does not consider one-way streets.  This is potentially
  dangerous is you blindly follow the route, hence you must confirm it
  is safe first.
* For some reason, OSM searches return all kinds of junk instead of
  administrative boundaries.  In these cases, you may need to used the
  --select option to pick the next search choice.
* Single threaded and slow.  For large areas, it may take 24-hours to
  work out the optimum route.

## Requirements

This repo hasn't been set up as a proper installation or package yet -
it's just a script and you'll need to sort out your own Python
dependencies.  You will need to pip install the following
packages (tested on late version of Python 3 only so far).

* shapely
* gpxpy
* numpy
* networkx
* geopandas
* osmnx

```bash
pip install -r requirements.txt
```

## Options

```bash
% ./optiburb.py
2020-07-28 21:13:26 optiburb.py:__init__:64 [WARNING] WARNING - this program does not consider the direction of one-way roads or other roads that may be not suitable for your mode of transport. You must confirm the path safe for yourself
usage: optiburb.py [-h] [--debug DEBUG] [--start START] [--prune] [--simplify] [--simplify-gpx] [--select SELECT] [--shapefile SHAPEFILE] [--buffer BUFFER] [--save-fig] [--save-boundary] ...

Optimum Suburb Route Generator

positional arguments:
  names                 suburb names with state, country, etc

optional arguments:
  -h, --help            show this help message and exit
  --debug DEBUG         debug level debug, info, warn, etc
  --start START         optional starting address
  --prune               prune unnamed gravel tracks
  --simplify            simplify OSM nodes on load
  --simplify-gpx        reduce GPX points
  --select SELECT       select the nth item from the search results. a truely awful hack because i cant work out how to search for administrative boundaries.
  --shapefile SHAPEFILE
                        filename of shapefile to load localities, comma separated by the column to match on
  --buffer BUFFER       buffer distsance around polygon
  --save-fig            save an SVG image of the nodes and edges
  --save-boundary       save a GPX file of the suburb boundary
% 
```


## Examples

To fetch data from OSM search using overpass API, state the long winded name of the suburb.

```bash
./optiburb.py "bellfield, victoria, australia"
```

You can add multiple adjoining suburbs and they will be merged
together (just incase a single suburb isn't big enough).

If the suburb fails with some weird message about no nodes in the
polygon, you may have selected a name, instead of the whole locality.
I haven't worked out how to specifically search for a administraive
boundary yet, so the ugly hack in the short term is to use --select 2
to pick the next search result.  Sometimes if you specify the entire
name including the broader admin boundary (shire, country, etc) it may
help.. You can experiment with the search at https://www.openstreetmap.org/

```bash
./optiburb.py --select 2 "footscray, victoria, australia"
```

You can also import polygon boundaries from shapefiles, but you'll
need to know the column name and the key in advance.  In this example,
the shapefile is from the Australian Government with state locality
boundaries.

```bash
./optiburb.py --save-fig --save-boundary --prune --shapefile ~/Projects/gis/VIC_LOCALITY_POLYGON_shp,vic_loca_2 KEW
```

Pruning a route will attempt to remove unnamed tracks, which tend to
be 4wd tracks and not a lot of fun to do on most bikes.

You save the polygon boundary as a GPX files, which is handy for
loading into head units to see where you're going.

You can also save the SVG node file, which shows the various
intersections and other OSM points.
