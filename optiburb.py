#!/usr/bin/env python3.9

# this is a undirected graph version.. one-way streets and multi-edges
# are reduced, which means:

# WARNING - the resulting paths are not guaranteed to be rideable or
# safe.  You must confirm the path yourself.

from burbing_new import Burbing
import logging
import argparse

logging.basicConfig(format='%(asctime)-15s %(filename)s:%(funcName)s:%(lineno)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

##
##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optimum Suburb Route Generator')
    parser.add_argument('names', type=str, nargs=argparse.REMAINDER, help='suburb names with state, country, etc')
    parser.add_argument('--debug', type=str, default='info', help='debug level debug, info, warn, etc')
    parser.add_argument('--start', type=str, help='optional starting address')
    parser.add_argument('--prune', default=False, action='store_true', help='prune unnamed gravel tracks')
    parser.add_argument('--simplify', default=False, action='store_true', help='simplify OSM nodes on load')
    parser.add_argument('--simplify-gpx', dest='simplify_gpx', default=True, action='store_true', help='reduce GPX points')
    parser.add_argument('--complex-gpx', dest='simplify_gpx', action='store_false', help='leave all the OSM points in the GPX output')
    parser.add_argument('--select', type=int, default=1, help='select the nth item from the search results. a truely awful hack because i cant work out how to search for administrative boundaries.')
    parser.add_argument('--shapefile', type=str, default=None, help='filename of shapefile to load localities, comma separated by the column to match on')
    parser.add_argument('--buffer', type=int, dest='buffer', default=20, help='buffer distsance around polygon')
    parser.add_argument('--save-fig', default=False, action='store_true', help='save an SVG image of the nodes and edges')
    parser.add_argument('--save-boundary', default=False, action='store_true', help='save a GPX file of the suburb boundary')
    parser.add_argument('--feature-deadend', default=False, action='store_true', help='experimental feature to optimised deadends in solution')

    args = parser.parse_args()

    log.setLevel(logging.getLevelName(args.debug.upper()))

    log.debug('called with args - %s', args)

 

    burbing = Burbing()
    burbing.main(args)


