#!/usr/bin/env python3.8

# this is a undirected graph version.. one-way streets and multi-edges
# are reduced, which means:

# WARNING - the resulting paths are not guaranteed to be rideable or
# safe.  You must confirm the path yourself.

import math
import time
import os
import sys
import re
import shapely
import logging
import geopandas
import osmnx
import networkx as nx
import numpy as np
import itertools
import argparse
import gpxpy
import gpxpy.gpx

# TODO - look into detecting private roads, surface select (gravel,
# etc).

logging.basicConfig(format='%(asctime)-15s %(filename)s:%(funcName)s:%(lineno)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)

class Burbing:

    WARNING = '''WARNING - this program does not consider the direction of one-way roads or other roads that may be not suitable for your mode of transport. You must confirm the path safe for yourself'''

    def __init__(self):

        self.g = None

        self.polygons = {}
        self.region = shapely.geometry.Polygon()
        self.name = ''
        self.start = None

        # XXX - should really import the default osmnx fitlers for
        # each activity type and adjust those.. The one below has a
        # parking_aisle filter added from the standard bike model.

        self.custom_filter = (
            f'["highway"]["area"!~"yes"]["highway"!~"footway|steps|corridor|elevator|'
            f'escalator|motor|proposed|construction|abandoned|platform|raceway"]'
            f'["bicycle"!~"no"]["service"!~"private|parking_aisle"]["access"!~"private"]'
        )


        # not all of this data is used but it looks interesting.

        useful_tags_way = [
            'bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name', 'highway', 'maxspeed', 'service',
            'access', 'area', 'landuse', 'width', 'est_width', 'junction', 'surface',
        ]

        osmnx.utils.config(useful_tags_way=useful_tags_way, use_cache=True, log_console=True)

        log.warning(self.WARNING)

        return

    ##
    ##
    def add_polygon(self, polygon, name):

        self.polygons[name] = polygon

        self.region = self.region.union(polygon)

        if self.name:
            self.name += '_'
            pass

        processed_name = name.lower()
        processed_name = re.sub(r'[\s+,]+', '_', processed_name)

        self.name += processed_name

        return

    ##
    ##
    def get_osm_polygon(self, name, select=1, buffer_dist=20):

        log.info('searching for query=%s, which_result=%s', name, select)

        gdf = osmnx.gdf_from_place(name, buffer_dist=buffer_dist, which_result=select)
        log.info('gdf=%s', gdf)

        polygon = gdf.geometry.values[0]

        return polygon

    ##
    ##
    def get_shapefile_polygon(self, shapefile, key, name):

        log.info('shapefile=%s, key=%s, name=%s', shapefile, key, name)

        df = shapefile

        suburb = df[df[key] == name]
        suburb = suburb.to_crs(epsg=4326)
        log.info('suburb=%s', suburb)

        polygon = suburb['geometry'].values[0]

        return polygon

    ##
    ##
    def set_start_location(self, addr):

        point =  osmnx.utils_geo.geocode(addr)
        self.start = point
        log.info('setting start point to %s', self.start)
        return

    ##
    ##
    def find_odd_nodes(self):

        # for undirected graphs

        odd_nodes = { i for i, n in self.g.degree if n % 2 == 1 }

        return odd_nodes

    ##
    ##
    def get_pair_combinations(self, nodes):
        pairs = list(itertools.combinations(nodes, 2))
        return pairs

    ##
    ##
    def get_shortest_path_pairs(self, g, pairs):

        shortest_paths = {}

        _cur_pct = 0
        _size = len(pairs)

        for n, pair in enumerate(pairs):
            i, j = pair
            shortest_paths[pair] = nx.dijkstra_path_length(g, i, j, weight='length')

            ## output progress

            _pct = int(100 * n / _size)
            if _pct != _cur_pct:
                _cur_pct = _pct
                log.debug('dijkstra progress %s%%, [%d/%d]', _pct, n, _size)
                pass
            pass

        return shortest_paths

    ##
    ##
    def augment_graph(self, g, pairs):

        graph_aug = g.copy()

        log.info('(augmented) eulerian=%s', nx.is_eulerian(graph_aug))

        for i, pair in enumerate(pairs):
            a, b = pair

            length, path = nx.single_source_dijkstra(g, a, b, weight='length')

            log.debug('PAIR[%s] nodes = (%s,%s), length=%s, path=%s', i, a, b, length, path)

            linestring = self.path_to_linestring(graph_aug, path)

            # create a linestring of paths...

            data = {
                'length': length,
                'augmented': True,
                'path': path,
                'geometry': linestring,
                'from': a,
                'to': b,
            }
            log.debug('  creating new edge (%s,%s) - data=%s', a, b, data)

            graph_aug.add_edge(a, b, **data)
            pass

        return graph_aug

    ##
    ##
    def determine(self):

        self.g_directed = self.g
        self.g = osmnx.utils_graph.get_undirected(self.g_directed)

        log.info('eulerian=%s', nx.is_eulerian(self.g))

        log.info('undirected eulerian=%s', nx.is_eulerian(self.g))

        #
        # find all odd nodes
        #
        # odds = { n for n, d in self.g.out_degree() if d % 2 == 1 and d != 1 }

        for edge in self.g.edges:
            data = self.g.get_edge_data(*edge, 0)

            _osmid = ','.join(data.get('osmid')) if type(data.get('osmid')) == list else str(data.get('osmid'))
            _name = ','.join(data.get('name')) if type(data.get('name')) == list else str(data.get('name'))
            _highway = data.get('highway', '-')
            _surface = data.get('surface', '-')
            _oneway = data.get('oneway', '-')
            _access = data.get('access', '-')
            log.debug(f'{_osmid:10} {_name:30} {_highway:20} {_surface:10} {_oneway:10} {_access:10}')
            pass

        self.odd_nodes = self.find_odd_nodes()

        log.info('odd_nodes=%s', len(self.odd_nodes))

        log.debug('calculating combinations')

        odd_node_pairs = self.get_pair_combinations(self.odd_nodes)

        log.info('combinations=%s', len(odd_node_pairs))

        #
        # get shortest path distances for all odd pairs
        #
        log.info('dijkstra path start for all combinations - this might take a long time.  Run with --debug=debug to see progress')
        odd_pair_paths = self.get_shortest_path_pairs(self.g, odd_node_pairs)

        # XXX - this part doesn't work well because it doesn't
        # consider the direction of the paths.

        # create another graph off odd pairs.. using negative weights
        # because we want minimum distances but only maximum algorithm
        # exists in networkx.
        #
        log.debug('new undirected graph of pairs with negative lengths')
        g_odd_nodes = nx.Graph()

        for k, length in odd_pair_paths.items():
            i,j = k
            g_odd_nodes.add_edge(i, j, **{'length': length, 'weight': -length})
            pass

        log.info('new nodes=%s, edges=%s', g_odd_nodes.order(), g_odd_nodes.size())

        log.info('(odd complete) eulerian=%s', nx.is_eulerian(g_odd_nodes))

        #
        # find min weight matching
        #
        log.info('calculating max weight matching - this can also take a while')

        odd_matching = nx.algorithms.max_weight_matching(g_odd_nodes, True)

        log.debug('odd_matching=%s', odd_matching)
        log.info('len=%s', len(odd_matching))

        #
        # augment original graph
        #
        log.info('augment original')

        graph_aug = self.augment_graph(self.g, odd_matching)

        start_node = self.get_start_node(self.g, self.start)

        log.info('(augmented) eulerian=%s', nx.is_eulerian(graph_aug))

        euler_circuit = list(nx.eulerian_circuit(graph_aug, source=start_node))

        results = []
        #log.debug('euler_circuit=%s', euler_circuit)

        return graph_aug, euler_circuit

        return

    ##
    ##
    def reverse_linestring(self, line):

        # surely there has to be a better built-in way to do this?

        coords = list(line.coords)[::-1]
        enil = shapely.geometry.LineString(coords)
        log.debug('reversing linestring line=%s', line)
        log.debug('  -> line=%s', enil)

        return enil

    ##
    ##
    def directional_linestring(self, g, edge):
        u, v = edge
        data = g.get_edge_data(u, v, 0)
        if data is None:
            log.error('no data for edge %s', edge)
            return None

        node_to = data.get('to')
        node_from = data.get('from')

        if (u, v) == (node_from, node_to):
            return data.get('geometry')

        if (u, v) == (node_to, node_from):
            return self.reverse_linestring(data.get('geometry'))

        log.error('failed to match start and end for directional linestring edge=%s, linestring=%s', edge, data)

        return None

    ##
    ##
    def get_start_node(self, g, start_addr):

        if start_addr:
            (start_node, distance) = osmnx.distance.get_nearest_node(g, start_addr, return_dist=True)
            log.info('start_node=%s, distance=%s', start_node, distance)
        else:
            start_node = None
            pass

        return start_node

    ##
    ##
    def path_to_linestring(self, g, path):

        coords = []
        prev = None

        u = path.pop(0)

        for v in path:

            edge = (u, v)

            log.debug('working with edge=%s', edge)

            data = g.get_edge_data(u, v, 0)

            if data is None:
                log.error('missing data for edge (%s, %s)', u, v)
                continue

            linestring = data.get('geometry')
            directional_linestring = self.directional_linestring(g, edge)

            for c in directional_linestring.coords:
                if c == prev: continue

                coords.append(c)
                prev = c
                pass

            u = v
            pass

        return shapely.geometry.LineString(coords)

    ##
    ##
    def prune(self):

        # eliminate edges with unnamed tracks..

        remove_types = ('track', 'path')

        removeset = set()
        for edge in self.g.edges:
            data = self.g.get_edge_data(*edge)

            if data.get('highway') in remove_types and data.get('name') is None:
                log.debug('removing edge %s, %s', edge, data)
                removeset.add(edge)
                pass

            if data.get('highway') in ('cycleway',):
                log.debug('removing edge %s, %s', edge, data)
                removeset.add(edge)
                pass
            pass

        for edge in removeset:
            self.g.remove_edge(*edge)
            pass

        # XXX - this might not work. the idea of pruning edge types by
        # tags after loading might not be right.. A better idea might
        # be to fiddle with filters before fetching the OSM data, or
        # look further into the features in osmnx to prune orphaned
        # nodes.

        self.g = osmnx.utils_graph.remove_isolated_nodes(self.g)
        return

    ##
    ##
    def save_fig(self):

        filepath = f'{self.name}-nodes.svg'

        log.info('saving SVG node file as %s', filepath)

        nc = ['red' if node in self.odd_nodes else 'blue' for node in self.g.nodes() ]

        fig, ax = osmnx.plot_graph(self.g, show=False, save=True, node_color=nc, filepath=filepath)

        return

    ##
    ##
    def load(self, options):

        log.info('fetching OSM data bounded by polygon')
        self.g = osmnx.graph_from_polygon(self.region, network_type='bike', simplify=False, custom_filter=self.custom_filter)

        log.debug('original g=%s, g=%s', self.g, type(self.g))
        log.info('original nodes=%s, edges=%s', self.g.order(), self.g.size())

        if options.simplify:
            log.info('simplifying graph')
            self.g = osmnx.simplification.simplify_graph(self.g, strict=False, remove_rings=False)
            pass

        return

    ##
    ##
    def load_shapefile(self, filename):

        df = geopandas.read_file(filename)
        log.info('df=%s', df)
        log.info('df.crs=%s', df.crs)

        return df

    ##
    ##
    def add_shapefile_region(self, name):

        df = self.shapefile_df
        key = self.shapefile_key

        suburb = df[df[key] == value]
        log.info('suburb=%s', suburb)
        suburb = suburb.to_crs(epsg=4326)
        log.info('suburb=%s', suburb)

        polygon = suburb['geometry'].values[0]

        return polygon

    ##
    ##
    def create_gpx_polygon(self, polygon):

        gpx = gpxpy.gpx.GPX()
        gpx.name = f'boundary {self.name}'
        gpx.author_name = 'optiburb'
        gpx.creator = 'experimental burbing'
        gpx.description = f'experimental burbing boundary for {self.name}'

        track = gpxpy.gpx.GPXTrack()
        track.name = f'burb bound {self.name}'

        filename = f'burb_polygon_{self.name}.gpx'

        log.info('saving suburb boundary - %s', filename)

        # XXX - add colour?

        #xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3"
        #track.extensions =
        #<extensions>
        #  <gpxx:TrackExtension>
        #    <gpxx:DisplayColor>Red</gpxx:DisplayColor>
        #  </gpxx:TrackExtension>
        #</extensions>

        gpx.tracks.append(track)

        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)

        for x, y in polygon.exterior.coords:
            segment.points.append(gpxpy.gpx.GPXRoutePoint(latitude=y, longitude=x))
            pass

        data = gpx.to_xml()
        with open(filename, 'w') as f:
            f.write(data)
            pass

        return

    ##
    ##
    def create_gpx_track(self, g, edges, simplify=False):

        # create GPX XML.

        gpx = gpxpy.gpx.GPX()
        gpx.name = f'burb {self.name}'
        gpx.author_name = 'optiburb'
        #gpx.author_email =''
        gpx.creator = 'experimental burbing'
        gpx.description = f'experimental burbing route for {self.name}'

        track = gpxpy.gpx.GPXTrack()
        track.name = f'burb trk {self.name}'
        gpx.tracks.append(track)

        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)

        #route.comment = ''
        #route.source = ''
        #route.name = ''

        distance = 0.0

        i = 1

        for n, edge in enumerate(edges):

            u, v = edge
            edge_data = g.get_edge_data(*edge, 0)
            #directed_edge_data = self.g_directed.get_edge_data(*edge, 0)

            log.debug('EDGE [%d] - edge=%s, data=%s', n, edge, edge_data)

            if edge_data is None:
                log.warning('null data for edge %s', edge)
                continue

            name = edge_data.get('name', '')
            _highway = edge_data.get('highway', '')
            linestring = edge_data.get('geometry')
            node_from = edge_data.get('from')
            node_to = edge_data.get('to')

            _surface = edge_data.get('surface', '')
            _oneway = edge_data.get('oneway', '')
            _access = edge_data.get('access', '')
            _distance = edge_data.get('length', 0)

            distance += _distance

            log.debug(' leg [%d] -> %s (%s,%s,%s,%s,%s)', n, name, _highway, _surface, _oneway, _access, _distance)

            #linestring = edge_data.get('geometry')
            augmented = edge_data.get('augmented')

            ## how do we know that we're going in the rigth direction??

            log.debug('  edge=%s, augmented=%s, from=%s, to=%s, linestring=%s', edge, augmented, node_from, node_to, linestring)

            if linestring:
                directional_linestring = self.directional_linestring(g, edge)

                for lon, lat in directional_linestring.coords:
                    segment.points.append(gpxpy.gpx.GPXRoutePoint(latitude=lat, longitude=lon))
                    log.debug('     INDEX[%d] = (%s, %s)', i, lat, lon)
                    i += 1
                    pass
                pass
            else:
                log.error('  no linestring for edge=%s', edge)
                #lat = g.nodes[u]['y']
                #lon = g.nodes[v]['x']
                #segment.points.append(gpxpy.gpx.GPXRoutePoint(latitude=lat, longitude=lon))
                pass

            #log.info('lat=%s, lon=%s, edge=%s, name=%s, highway=%s', lat, lon, edge, name, highway)

            pass

        log.info('total distance = %.1fkm', distance/1000.0)

        ##
        ##
        if simplify:
            log.info('simplifying GPX')
            gpx.simplify()
            pass

        data = gpx.to_xml()
        filename = f'burb_track_{self.name}.gpx'

        with open(filename, 'w') as f:
            f.write(data)
            pass

        return

    pass

##
##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optimum Suburb Route Generator')
    parser.add_argument('names', type=str, nargs=argparse.REMAINDER, help='suburb names with state, country, etc')
    parser.add_argument('--debug', type=str, default='info', help='debug level debug, info, warn, etc')
    parser.add_argument('--start', type=str, help='optional starting address')
    parser.add_argument('--prune', default=False, action='store_true', help='prune unnamed gravel tracks')
    parser.add_argument('--simplify', default=False, action='store_true', help='simplify OSM nodes on load')
    parser.add_argument('--simplify-gpx', default=False, action='store_true', help='reduce GPX points')
    parser.add_argument('--select', type=int, default=1, help='select the nth item from the search results. a truely awful hack because i cant work out how to search for administrative boundaries.')
    parser.add_argument('--shapefile', type=str, default=None, help='filename of shapefile to load localities, comma separated by the column to match on')
    parser.add_argument('--buffer', type=int, dest='buffer', default=20, help='buffer distsance around polygon')
    parser.add_argument('--save-fig', default='False', action='store_true', help='save an SVG image of the nodes and edges')
    parser.add_argument('--save-boundary', default='False', action='store_true', help='save a GPX file of the suburb boundary')

    args = parser.parse_args()

    log.setLevel(logging.getLevelName(args.debug.upper()))

    log.debug('called with args - %s', args)

    burbing = Burbing()

    if not args.names:
        parser.print_help()
        sys.exit(1)
        pass

    if args.shapefile:

        filename, key = args.shapefile.split(',')

        log.info('shapefile=%s, key=%s', filename, key)

        shapefile = burbing.load_shapefile(filename)

        for name in args.names:
            polygon = burbing.get_shapefile_polygon(shapefile, key, name)
            burbing.add_polygon(polygon, name)
            pass
        pass

    else:

        for name in args.names:

            polygon = burbing.get_osm_polygon(name, args.select, args.buffer)
            burbing.add_polygon(polygon, name)
            pass
        pass

    if args.save_boundary:
        burbing.create_gpx_polygon(burbing.region)
        pass

    if args.start:
        burbing.set_start_location(args.start)
        pass

    burbing.load(args)

    if args.prune:
        burbing.prune()
        pass

    graph_aug, euler_circuit = burbing.determine()
    burbing.create_gpx_track(graph_aug, euler_circuit, args.simplify)

    if args.save_fig:
        burbing.save_fig()
        pass

    pass

