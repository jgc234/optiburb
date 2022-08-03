import time
import sys
import re
import shapely
import logging
import geopandas
import osmnx
import networkx as nx
import itertools
import gpxpy
import gpxpy.gpx
import datetime

logging.basicConfig(format='%(asctime)-15s %(filename)s:%(funcName)s:%(lineno)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

class Burbing:

    WARNING = '''WARNING - this program does not consider the direction of one-way roads or other roads that may be not suitable for your mode of transport. You must confirm the path safe for yourself'''

    def __init__(self):

        self.g = None

        self.polygons = {}
        self.region = shapely.geometry.Polygon()
        self.name = ''
        self.start = None

        #
        # filters to roughly match those used by rendrer.earth (see
        # https://wandrer.earth/scoring )
        #
        self.custom_filter = (

            '["highway"]'

            '["area"!~"yes"]'

            #'["highway"!~"motorway|motorway_link|trunk|trunk_link|bridleway|footway|service|pedestrian|'
            '["highway"!~"motorway|motorway_link|bridleway|footway|service|pedestrian|'
            'steps|stairs|escalator|elevator|construction|proposed|demolished|escape|bus_guideway|'
            'sidewalk|crossing|bus_stop|traffic_signals|stop|give_way|milestone|platform|speed_camera|'
            'raceway|rest_area|traffic_island|services|yes|no|drain|street_lamp|razed|corridor|abandoned"]'

            '["access"!~"private|no|customers"]'

            '["bicycle"!~"dismount|use_sidepath|private|no"]'

            '["service"!~"private|parking_aisle"]'

            '["motorroad"!="yes"]'

            '["golf_cart"!~"yes|designated|private"]'

            '[!"waterway"]'

            '[!"razed"]'
        )


        log.debug('custom_filter=%s', self.custom_filter)

        # not all of these fields are used at the moment, but they
        # look like fun for the future.

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
        processed_name = re.sub(r'[\s,_]+', '_', processed_name)

        self.name += processed_name

        return

    ##
    ##
    def get_osm_polygon(self, name, select=1, buffer_dist=20):

        log.info(f'searching for query={name}, which_result={select}')

        gdf = osmnx.geocode_to_gdf(name, buffer_dist=buffer_dist, which_result=select)
        log.info(f'gdf={gdf}')

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

        point =  osmnx.geocoder.geocode(addr)
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

        # XXX - consider Floyd–Warshall here instead of repeated
        # Dijkstra.  Also consider how to parallelise this as a
        # short-term speed-up, by palming off chunks to another
        # thread, except this wont work in python.

        shortest_paths = {}

        _prev_pct = 0
        _size = len(pairs)
        _prev_n = 0
        _prev_time = time.time()

        for n, pair in enumerate(pairs):
            i, j = pair
            shortest_paths[pair] = nx.dijkstra_path_length(g, i, j, weight='length')

            # ## output progress

            _cur_pct = int(100 * n / _size)
            if _prev_pct != _cur_pct:
                _cur_time = time.time()
                log.info('dijkstra progress %s%%, [%d/%d] %d/second', _cur_pct, n, _size, (_prev_n - n) / (_prev_time - _cur_time))

                _prev_time = _cur_time
                _prev_pct = _cur_pct
                _prev_n = n
                pass
            pass

        return shortest_paths

    ##
    ##
    def augment_graph(self, pairs):

        # create a new graph and stuff in the new fake/virtual edges
        # between odd pairs.  Generate the edge metadata to make them
        # look similar to the native edges.

        log.info('pre augmentation eulerian=%s', nx.is_eulerian(self.g_augmented))

        for i, pair in enumerate(pairs):
            a, b = pair

            length, path = nx.single_source_dijkstra(self.g, a, b, weight='length')

            log.debug('PAIR[%s] nodes = (%s,%s), length=%s, path=%s', i, a, b, length, path)

            linestring = self.path_to_linestring(self.g_augmented, path)

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

            self.g_augmented.add_edge(a, b, **data)
            pass

        log.info('post augmentation eulerian=%s', nx.is_eulerian(self.g_augmented))

        return


    ##
    ##
    def print_edges(self, g):

        for edge in g.edges:
            data = g.get_edge_data(*edge, 0)

            _osmid = ','.join(data.get('osmid')) if type(data.get('osmid')) == list else str(data.get('osmid'))
            _name = ','.join(data.get('name')) if type(data.get('name')) == list else str(data.get('name'))
            _highway = data.get('highway', '-')
            _surface = data.get('surface', '-')
            _oneway = data.get('oneway', '-')
            _access = data.get('access', '-')
            log.debug(f'{_osmid:10} {_name:30} {_highway:20} {_surface:10} {_oneway:10} {_access:10}')
            pass

    ##
    ##
    def determine_nodes(self):

        self.g_directed = self.g
        self.g = osmnx.utils_graph.get_undirected(self.g_directed)

        self.print_edges(self.g)

        self.g_augmented = self.g.copy()

        self.odd_nodes = self.find_odd_nodes()

        return


    ##
    ##
    def optimise_dead_ends(self):

        # preempt virtual path augmentation for the case of a dead-end
        # road.  Here the optimum combination pair is its only
        # neighbour node, so why bother iterating through all the
        # pairs to find that.

        # XXX - not quite clean yet.. we are augmenting the original
        # grpah.. need a cleaner way to pass changes through the
        # processing pipeline.

        deadends = { i for i, n in self.g.degree if n == 1 }

        n1 = len(self.find_odd_nodes())

        for deadend in deadends:

            neighbours = self.g[deadend]

            #node_data = self.g.nodes[deadend]
            #log.info('deadend_ndoe=%s, data=%s', deadend, node_data)
            log.debug('deadend_node=%s', deadend)

            if len(neighbours) != 1:
                log.error('wrong number of neighbours for a dead-end street')
                continue

            for neighbour in neighbours.keys():
                log.debug('neighbour=%s', neighbour)

                edge_data = dict(self.g.get_edge_data(deadend, neighbour, 0))
                edge_data['augmented'] = True
                
                log.debug('  creating new edge (%s,%s) - data=%s', deadend, neighbour, edge_data)

                self.g.add_edge(deadend, neighbour, **edge_data)

                pass

            pass

        # fix up the stuff we just busted.  XXX - this should not be
        # hidden in here.

        self.odd_nodes = self.find_odd_nodes()
        self.g_augmented = self.g.copy()

        n2 = len(self.odd_nodes)

        log.info('odd_nodes_before=%d, odd_nodes_after=%d', n1, n2)
        log.info('optimised %d nodes out', n1 - n2)

        return

    ##
    ##
    def determine_combinations(self):

        log.info(f'eulerian=%s, odd_nodes=%s', nx.is_eulerian(self.g), len(self.odd_nodes))

        odd_node_pairs = self.get_pair_combinations(self.odd_nodes)

        log.info('combinations=%s', len(odd_node_pairs))

        odd_pair_paths = self.get_shortest_path_pairs(self.g, odd_node_pairs)

        # XXX - this part doesn't work well because it doesn't
        # consider the direction of the paths.

        # create a temporary graph of odd pairs.. really we should be
        # doing the combination max calculations here.

        self.g_odd_nodes = nx.Graph()

        for k, length in odd_pair_paths.items():
            i,j = k
            attrs = {
                'length': length,
                'weight': -length,
            }

            self.g_odd_nodes.add_edge(i, j, **attrs)
            pass

        log.info('new_nodes=%s, edges=%s, eulerian=%s', self.g_odd_nodes.order(), self.g_odd_nodes.size(), nx.is_eulerian(self.g_odd_nodes))

        log.info('calculating max weight matching - this can also take a while')

        return


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

        # this creates a new linestring that follows the path of the
        # augmented route between two odd nodes.  this is needed to
        # force a path with the final GPX route, rather than drawing a
        # straight line between the two odd nodes and hoping some
        # other program route the same way we wanted to.

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

        # eliminate edges with unnamed tracks.  At least where I live,
        # these tend to be 4wd tracks that require a mountain bike to
        # navigate.  probably need to do a better fitler that looks at
        # surface type and other aspects.

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

        # this removes the isolated nodes orphaned from the removed
        # edges above.  It does not solve the problem of a
        # non-connected graph (ie, nodes and edges in a blob that
        # aren't reachable to other parts of the graph)

        self.g = osmnx.utils_graph.remove_isolated_nodes(self.g)
        return

    ##
    ##
    def save_fig(self):

        filename = f'burb_nodes_{self.name}.svg'

        log.info('saving SVG node file as %s', filename)

        nc = ['red' if node in self.odd_nodes else 'blue' for node in self.g.nodes() ]

        fig, ax = osmnx.plot_graph(self.g, show=False, save=True, node_color=nc, filepath=filename)

        return

    ##
    ##
    def load(self, options=None):

        log.info('fetching OSM data bounded by polygon')
        self.g = osmnx.graph_from_polygon(self.region, network_type='bike', simplify=False, custom_filter=self.custom_filter)

        log.debug('original g=%s, g=%s', self.g, type(self.g))
        log.info('original nodes=%s, edges=%s', self.g.order(), self.g.size())


        if options is None: return

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

        stats_distance = 0.0
        stats_backtrack = 0.0
        stats_deadends = 0

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

        i = 1

        for n, edge in enumerate(edges):

            u, v = edge
            edge_data = g.get_edge_data(*edge, 0)

            log.debug('EDGE [%d] - edge=%s, data=%s', n, edge, edge_data)

            if edge_data is None:
                log.warning('null data for edge %s', edge)
                continue

            linestring = edge_data.get('geometry')
            augmented = edge_data.get('augmented')
            stats_distance += edge_data.get('length', 0)

            log.debug(' leg [%d] -> %s (%s,%s,%s,%s,%s)', n, edge_data.get('name', ''), edge_data.get('highway', ''), edge_data.get('surface', ''), edge_data.get('oneway', ''), edge_data.get('access', ''), edge_data.get('length', 0))

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
                pass

            if edge_data.get('augmented', False):
                stats_backtrack += edge_data.get('length', 0)
                pass

            pass

        log.info('total distance = %.2fkm', stats_distance/1000.0)
        log.info('backtrack distance = %.2fkm', stats_backtrack/1000.0)
        
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
    
    def main(self, args):
        start_time = datetime.datetime.now()
            
        if not args.names:
            parser.print_help()
            sys.exit(1)
            pass

        if args.shapefile:
            # If a shapefile is given

            filename, key = args.shapefile.split(',')

            log.info('shapefile=%s, key=%s', filename, key)

            shapefile = self.load_shapefile(filename)

            for name in args.names:
                polygon = self.get_shapefile_polygon(shapefile, key, name)
                self.add_polygon(polygon, name)
                pass
            pass

        else:
            # If a location name is given
            for name in args.names:
                polygon = self.get_osm_polygon(name, args.select, args.buffer)
                self.add_polygon(polygon, name)
                pass

            pass

        if args.save_boundary:
            self.create_gpx_polygon(self.region)
            pass

        if args.start:
            self.set_start_location(args.start)
            pass

        self.load(args)

        if args.prune:
            self.prune()
            pass

        self.determine_nodes()

        if args.feature_deadend:
            self.optimise_dead_ends()
            pass

        if args.save_fig:
            self.save_fig()
            pass

        self.determine_combinations()

        self.determine_circuit()

        self.create_gpx_track(self.g_augmented, self.euler_circuit, args.simplify_gpx)

        end_time = datetime.datetime.now()

        log.info('elapsed time = %s', end_time - start_time)

        pass

