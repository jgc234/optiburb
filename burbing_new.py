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
import pulp
import multiprocessing as mp 
from multiprocessing import Process, Manager

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
    def get_directed_degrees(self, graph):
        return {i: out_deg - in_deg for (i, out_deg),(_,in_deg) in
                zip(graph.out_degree, graph.in_degree)}

    def find_unbalanced_nodes(self):
        self.directed_degrees = self.get_directed_degrees(self.g)

        self.unbalanced_nodes = [i for i, deg in self.directed_degrees.items() if deg != 0]
        self.unbalanced_in_nodes = [i for i, deg in self.directed_degrees.items() if deg < 0]
        self.unbalanced_out_nodes = [i for i, deg in self.directed_degrees.items() if deg > 0]

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

            ## output progress

            # _cur_pct = int(100 * n / _size)
            # if _prev_pct != _cur_pct:
            #     _cur_time = time.time()
            #     log.info('dijkstra progress %s%%, [%d/%d] %d/second', _cur_pct, n, _size, (_prev_n - n) / (_prev_time - _cur_time))
            #
            #     _prev_time = _cur_time
            #     _prev_pct = _cur_pct
            #     _prev_n = n
            #     pass
            # pass

        return shortest_paths

    ##
    ##
    def augment_graph(self):
        # create a new graph and stuff in the new fake/virtual edges
        # between odd pairs.  Generate the edge metadata to make them
        # look similar to the native edges.
        self.g_augmented = self.g.copy()

        for edge, mult in self.solution.items():
            i,j = edge
            print(edge)
            _, path = nx.single_source_dijkstra(self.g, i, j, weight='length')
            for (a,b) in zip(path,path[1:]):
                new_edge = self.g.edges.get((a,b,0))
                for _ in range(mult):
                    self.g_augmented.add_edge(a,b,**new_edge, augmented=True)


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
    ##
    ##
    # def optimise_dead_ends(self):
    #
    #     # preempt virtual path augmentation for the case of a dead-end
    #     # road.  Here the optimum combination pair is its only
    #     # neighbour node, so why bother iterating through all the
    #     # pairs to find that.
    #
    #     # XXX - not quite clean yet.. we are augmenting the original
    #     # grpah.. need a cleaner way to pass changes through the
    #     # processing pipeline.
    #
    #     deadends = { i for i, n in self.g.degree if n == 1 }
    #
    #     n1 = len(self.find_odd_nodes())
    #
    #     for deadend in deadends:
    #
    #         neighbours = self.g[deadend]
    #
    #         #node_data = self.g.nodes[deadend]
    #         #log.info('deadend_ndoe=%s, data=%s', deadend, node_data)
    #         log.debug('deadend_node=%s', deadend)
    #
    #         if len(neighbours) != 1:
    #             log.error('wrong number of neighbours for a dead-end street')
    #             continue
    #
    #         for neighbour in neighbours.keys():
    #             log.debug('neighbour=%s', neighbour)
    #
    #             edge_data = dict(self.g.get_edge_data(deadend, neighbour, 0))
    #             edge_data['augmented'] = True
    #             
    #             log.debug('  creating new edge (%s,%s) - data=%s', deadend, neighbour, edge_data)
    #
    #             self.g.add_edge(deadend, neighbour, **edge_data)
    #
    #             pass
    #
    #         pass
    #
    #     # fix up the stuff we just busted.  XXX - this should not be
    #     # hidden in here.
    #
    #     self.odd_nodes = self.find_odd_nodes()
    #     self.g_augmented = self.g.copy()
    #
    #     n2 = len(self.odd_nodes)
    #
    #     log.info('odd_nodes_before=%d, odd_nodes_after=%d', n1, n2)
    #     log.info('optimised %d nodes out', n1 - n2)
    #
    #     return
    def compute_distance(self, graph, node, distances):
        i,j = node
        try:
            d = nx.dijkstra_path_length(graph, i, j, weight="length")
            distances[(i,j)] = (d)
        except nx.NetworkXNoPath:
            distances[(i,j)] = (1e10)

    def compute_distances(self, graph, unbalanced_in_nodes, unbalanced_out_nodes):
        distances = {}
        with Manager() as manager:
            d = manager.dict()
            with manager.Pool(processes=8) as pool:
                pool.starmap(self.compute_distance, zip(itertools.repeat(graph),
                    itertools.product(unbalanced_in_nodes, unbalanced_out_nodes), itertools.repeat(d)))
            distances = dict(d)
        return distances


    def solve_optimization_problem(self):
        lp_problem = pulp.LpProblem("Problem", pulp.LpMinimize)
        
        #Get problem variables
        variables = {pulp.LpVariable(str(i) + "_" + str(j), lowBound=0, cat='Integer'):(i,j)
                for i in self.unbalanced_in_nodes for j in self.unbalanced_out_nodes}

        distances = self.compute_distances(self.g, self.unbalanced_in_nodes,
                self.unbalanced_out_nodes)

        #Optimization function
        lp_problem += pulp.lpSum([var*distances[node] for var, node in variables.items()])


        #Constraints
        for i in self.unbalanced_in_nodes:
            lp_problem += pulp.lpSum([var for var, (i2,_) in variables.items() if i2==i]) == -self.directed_degrees[i]

        
        for j in self.unbalanced_out_nodes:
            lp_problem += pulp.lpSum([var for var, (_,j2) in variables.items() if j2==j]) == self.directed_degrees[j]


        lp_problem.solve()
        log.info(f"Optimization Problem Status: {pulp.LpStatus[lp_problem.status]}")

        self.solution = {variables[v]:int(v.varValue) for v in lp_problem.variables() if v.varValue>0}


        log.info(f"Optimization Problem Solution Extra Distance: {sum(distances[key] for key in self.solution.keys())}")

    ##
    ##
    def determine_circuit(self):
        log.info(f"Starging with eulerian={nx.is_eulerian(self.g)}")

        self.find_unbalanced_nodes()

        log.info(f"Found {len(self.unbalanced_nodes)} unbalanced nodes")

        self.solve_optimization_problem()

        self.augment_graph()

        log.info(f'Post augmentation eulerian={nx.is_eulerian(self.g_augmented)}')

        start_node = self.get_start_node(self.g, self.start)

        self.euler_circuit = list(nx.eulerian_circuit(self.g_augmented, source=start_node))

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
        self.g_initial = self.g

        log.debug('original g=%s, g=%s', self.g, type(self.g))
        log.info('original nodes=%s, edges=%s', self.g.order(), self.g.size())


        #Get only largest strongly connected component
        scc = max(nx.strongly_connected_components(self.g), key=len)
        self.g = nx.MultiDiGraph(self.g.subgraph(scc))

        # if options.simplify:
        #     log.info('simplifying graph')
        #     self.g = osmnx.simplification.simplify_graph(self.g, strict=False, remove_rings=False)
        #     pass

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
        log.info(g.edges(data=True))

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

            start_node, _ = edge
            edge_data = g.get_edge_data(*edge, 0)

            log.debug('EDGE [%d] - edge=%s, data=%s', n, edge, edge_data)

            if edge_data is None:
                log.warning('null data for edge %s', edge)
                continue

            stats_distance += edge_data.get('length', 0)

            log.debug(' leg [%d] -> %s (%s,%s,%s,%s,%s)', n, edge_data.get('name', ''), edge_data.get('highway', ''), edge_data.get('surface', ''), edge_data.get('oneway', ''), edge_data.get('access', ''), edge_data.get('length', 0))

            lon = g.nodes.get(start_node)['x']
            lat = g.nodes.get(start_node)['y']

            segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon))
            log.debug('     INDEX[%d] = (%s, %s)', i, lat, lon)
            i += 1

            if edge_data.get('augmented', False):
                log.info(edge)
                stats_backtrack += edge_data.get('length', 0)

        log.info('total distance = %.2fkm', stats_distance/1000.0)
        log.info('backtrack distance = %.2fkm', stats_backtrack/1000.0)

        ##
        ##
        if simplify:
            # log.info('simplifying GPX')
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

        # if args.feature_deadend:
        #     self.optimise_dead_ends()
        #     pass
        #
        # if args.save_fig:
        #     self.save_fig()
        #     pass

        self.determine_circuit()

        self.create_gpx_track(self.g_augmented, self.euler_circuit, args.simplify_gpx)

        end_time = datetime.datetime.now()

        log.info('elapsed time = %s', end_time - start_time)

        pass

