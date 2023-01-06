import sys
import os
import math
import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from complex_split import complex_split
import logging

DATA_DIR = "data"
LOG_LEVEL = "WARNING"
YEAR = 2564
ROUGHNESS_RATE = 1.0
MAX_DM_DISTANCE = 1600            # meters
DEFAULT_DM_VALVE_SIZE = 300       # mm
DEFAULT_RESV_VALVE_SIZE = 300     # mm
DEFAULT_PIPE_DIAMETER_SIZE = 1000 # mm
SNAP_THRESHOLD = 0.5              # meters
VALVE_ON_MAIN_THRESHOLD = 10      # meters
NODE_COORDS_DECIMAL_PLACES = 2    # decimal places
SNAP_PIPE_LENGTH_THRESHOLD = 0.5  # meters
SNAP_PIPE_MAX_DISTANCE = 0.5      # meters

try:
    from config import *
except ImportError:
    pass

PREFIX = Path(DATA_DIR)
CONFIG = {
    "1A-1": PREFIX / "1A-1 Trunk Main GIS" / "MainPipe.shp",
    "1A-2": PREFIX / "1A-2 Administrative area GIS" / "DMA.shp",
    "1A-3": PREFIX / "1A-3 UUZP GIS" / "UUZP.shp",
    "1B"  : PREFIX / "1B Pipe characteristics" / "Pipe characteristics.csv",
    "2A"  : PREFIX / "2A DM GIS" / "DM_GIS.shp",
    "2B-1": PREFIX / "2B-1 DM Default Flow" / "Default_DM_Flow.csv",
    "2B-2": PREFIX / "2B-2 DM Flow Pattern" / "DM_Flow_Pattern.csv",
    "2B-3": PREFIX / "2B-3 DM Pressure Pattern" / "DM_Pressure_Pattern.csv",
    "3A"  : PREFIX / "3A Reservoir and Inlet GIS" / "3A Reservoir and Inlet GIS.shp",
    "3B-1": PREFIX / "3B-1 Reservoir and Inlet Characteristic" / "Reservoir Characteristic.csv",
    "3B-2": PREFIX / "3B-2 Reservoir and Inlet Patterns" / "Reservoir_Pattern_Oct2021.csv",
    "4A"  : PREFIX / "4A Valve on Main GIS" / "Valve_SettingMAP.shp",
    "4B"  : PREFIX / "Valve Characteristics and Valve Control" / "Valve Characteristic (4B-1).txt",
}

ATTR_MAPPING = {
    "1A-1": "gis_trunkmain",
    "1A-2": "gis_dma",
    "1A-3": "gis_uuz",
    "1B"  : "df_pipechar",
    "1B-2": "emitter",
    "2A"  : "gis_dm",
    "2B-1": "df_f_avg",
    "2B-2": "df_f_pattern",
    "2B-3": "df_p_pattern",
    "3A"  : "gis_resv",
    "3B-1": "df_resv_type",
    "3B-2": "df_resv_pattern",
    "4A"  : "gis_valve",
    "4B"  : "valve_char",
}

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter("%(levelname)s: %(message)s")
log_file_handler = logging.FileHandler('epanet_prep.log')
log_console_handler = logging.StreamHandler()
log_file_handler.setFormatter(log_formatter)
log_console_handler.setFormatter(log_formatter)
logger.addHandler(log_file_handler)
logger.addHandler(log_console_handler)
logger.setLevel(LOG_LEVEL)


@dataclass
class Emitter:
    year: int
    age: int
    module: str
    c1: float
    c2: float


@dataclass
class Node:
    id: str
    type: str
    geometry: LineString


@dataclass
class Pipe:
    id: str
    node1: Node
    node2: Node
    type: str   # main or valve
    diameter: float
    roughness: float
    minorloss: float
    row: int    # one-based row number from original GIS data
    status: str
    comment: str
    geometry: LineString
    flag: int = 0


@dataclass
class Junction:
    node: Node
    elev: float
    demand: float
    pattern: str
    comment: str


@dataclass
class Vertex:
    pipe: Pipe
    x: float
    y: float


class Epanet:

    def __init__(self, from_pickle_file=None):
        if from_pickle_file:
            self.load(from_pickle_file)
        else:
            self.pipes = {}
            self.nodes = {}
            self.vertices = []
            self.junctions = []
            self._node_counter = 0
            self._node_map_by_coords = {}
            self._pipe_map = {}  # map pipe's original zero-based index to list of derived pipes
        self.output_stream = sys.stdout

    def read_inputs(self):
        global scraped_dm
        """Read all input data from the global configuration CONFIG."""

        for a in ATTR_MAPPING.values():
            setattr(self, a, None)

        logger.info("Reading trunkmain GIS")
        df = gpd.read_file(CONFIG["1A-1"])
        self.adjust_df(df)
        setattr(self, ATTR_MAPPING['1A-1'], df)

        logger.info("Reading DMA GIS")
        df = gpd.read_file(CONFIG["1A-2"], encoding="cp874")
        self.adjust_df(df)
        setattr(self, ATTR_MAPPING['1A-2'], df)

        logger.info("Reading U/UZ GIS")
        #df = gpd.read_file(CONFIG["1A-3"], encoding="cp874")
        df = gpd.read_file(CONFIG["1A-3"], encoding="utf-8")
        self.adjust_df(df)
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING['1A-3'], df)

        logger.info("Reading pipe characteristics")
        df = pd.read_csv(CONFIG["1B"])
        self.adjust_df(df)
        df.set_index(['matl', 'pipe_size'], inplace=True)
        setattr(self, ATTR_MAPPING['1B'], df)

        logger.info("Reading DM GIS")
        df = gpd.read_file(CONFIG["2A"], encoding="cp874")
        self.adjust_df(df)
        df.drop_duplicates(subset=['name'], inplace=True)
        df.set_index('name', inplace=True)
        scraped_dm = gpd.read_file('DM_Merge2_Edit.geojson')
        scraped_dm.to_crs("EPSG:32647", inplace=True)
        scraped_dm.drop_duplicates(subset=['dma_name'], inplace=True)
        # rename DM-xx-xx-xx-xx to xxxxxxxx
        scraped_dm['name'] = scraped_dm.dma_name.apply(lambda s: s.replace('-', '')[2:])
        scraped_dm.set_index('name', inplace=True)
        scraped_dm['pipe_size'] = df.pipe_size
        scraped_dm.loc[scraped_dm['pipe_size'].astype(str)==str(np.nan),'pipe_size']=300
        # scraped_dm.to_csv (r'./export_dataframe.csv', index = False, header=True)
        # print (scraped_dm)
        setattr(self, ATTR_MAPPING['2A'], scraped_dm)

        df = pd.read_csv(CONFIG['2B-1'])
        df.columns = ['name', 'base']
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING['2B-1'], df)

        logger.info("Reading DM flow pattern")
        df = pd.read_csv(CONFIG['2B-2'], usecols=range(27), encoding='cp874')
        df.columns = ['name', 'base'] + [f'h{h}' for h in range(24)] + ['remark']
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING['2B-2'], df)

        logger.info("Reading DM pressure pattern")
        df = pd.read_csv(CONFIG['2B-3'])
        df.columns = ['name'] + [f'h{h}' for h in range(24)]
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING['2B-3'], df)

        #logger.info("Reading PRV control")
        #with open(CONFIG["2B-2"]) as f:
        #    prv = f.read()
        #setattr(self, ATTR_MAPPING['2B-2'], prv)

        logger.info("Reading reservoir GIS")
        df = gpd.read_file(CONFIG["3A"], encoding="cp874")
        self.adjust_df(df)
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING["3A"], df)

        logger.info("Reading reservoir types")
        df = pd.read_csv(CONFIG["3B-1"], header=None, encoding="cp874")
        df.columns = ['name', 'type', 'demand_or_head', 'name2']
        df.drop('name2', axis='columns', inplace=True)
        df.drop(df[pd.isna(df.name)].index, inplace=True)
        for idx, row in df[df.duplicated('name')].iterrows():
            logger.warning(f"Found duplicate entry for {row['name']}")
        df.drop_duplicates('name', inplace=True)
        df.set_index('name', inplace=True)
        df.demand_or_head = df.demand_or_head.astype(float)
        setattr(self, ATTR_MAPPING['3B-1'], df)

        logger.info("Reading reservoir patterns")
        df = pd.read_csv(CONFIG["3B-2"], header=None, encoding="cp874")
        df.columns = ['name', 'type', 'average'] + [f'h{h}' for h in range(24)]
        df.drop(df[pd.isna(df.name)].index, inplace=True)
        df.set_index('name', inplace=True)
        setattr(self, ATTR_MAPPING['3B-2'], df)

        logger.info("Reading valves on trunkmain")
        df = gpd.read_file(CONFIG['4A'], encoding="utf8").to_crs("EPSG:32647")
        self.adjust_df(df)
        setattr(self, ATTR_MAPPING['4A'], df)

        # XXX how to read valves' characteristics

    @staticmethod
    def adjust_df(df):
        """
        Rename all the columns in the provided dataframe to lowercase
        """
        df.columns = [c.lower() for c in df.columns]
        #df.index = np.arange(1, len(df) + 1)

    @staticmethod
    def read_emitter(fname):
        with open(fname) as f:
            y, a, m, c1, c2 = f.read().split(",")
            y = int(y)
            a = int(a)
            c1 = float(c1)
            c2 = float(c2)
            return Emitter(y, a, m, c1, c2)

    def output(self, s=""):
        print(s, file=self.output_stream)

    def get_coords_map_key(self, point):
        x, y = point.coords[0]
        kx = round(x*(10**NODE_COORDS_DECIMAL_PLACES))
        ky = round(y*(10**NODE_COORDS_DECIMAL_PLACES))
        assert type(kx) is int
        assert type(ky) is int
        return (kx, ky)

    def get_or_create_node(self, point, always_create=False):
        """
        Get or create a node object and update the node database.  If the
        point is close to an existing node, return the node (unless it is a DM
        or a reservoir); otherwise, create an return a new node with a unique
        ID.  The unique ID is obtained from a running number.

        Matching is done with coordinate values rounded to two decimal places
        for now.  In the future, a threshold may be specified to find a node
        that is considered close.

        TODO: consider using scipy's spatial index
        REF: https://gis.stackexchange.com/a/301935
        """
        coords_key = self.get_coords_map_key(point)
        if not always_create:
            try:
                node = self._node_map_by_coords[coords_key]
                if node.type in ['dm', 'resv']:
                    create = True
                else:
                    create = False
            except KeyError:
                create = True
        else:
            create = True

        if create:
            self._node_counter += 1
            node = Node(str(self._node_counter), None, point)
            node.ref = None   # to be replaced with reference to a dataframe
            node.pipes = []
            # TODO coords_key may overwrite existing entry
            self._node_map_by_coords[coords_key] = node
            self.nodes[node.id] = node

        return node

    def update_node_id(self, node, new_id, avoid_duplicate):
        """
        Update the id of the specified node with new_id, as well as updating
        the corresponding dict key.
        """
        if avoid_duplicate:
            new_id = self.find_unique_node_id(new_id)
        else:
            if new_id in self.nodes:
                raise ValueError(f'Node {new_id} already exists')
        logger.debug(f'Rename node {node.id} to {new_id}')
        node.original_id = node.id
        node.id = new_id
        self.nodes[new_id] = self.nodes.pop(node.original_id)
        return new_id

    def update_pipe_id(self, pipe, new_id, avoid_duplicate):
        """
        Update the id of the specified pipe with new_id, as well as updating
        the corresponding dict key.
        """
        if avoid_duplicate:
            new_id = self.find_unique_pipe_id(new_id)
        else:
            if new_id in self.pipes:
                raise ValueError(f'Pipe {new_id} already exists')
        logger.debug(f'Rename pipe {pipe.id} to {new_id}')
        id = pipe.id
        pipe.id = new_id
        self.pipes[new_id] = self.pipes.pop(id)
        return new_id

    def find_unique_node_id(self, basename):
        while basename in self.nodes:
            basename += 'x'
        return basename

    def find_unique_pipe_id(self, basename):
        while basename in self.pipes:
            basename += 'x'
        return basename

    def split_pipe(self, pipe, point):
        """Split the pipe at the specified point and update the pipe database"""
        x, y = point.coords[0][:]
        hsplitter = LineString([(x-1e-8, y), (x+1e-8, y)])
        vsplitter = LineString([(x, y-1e-8), (x, y+1e-8)])
        if pipe.geometry.crosses(hsplitter):
            splitter = hsplitter
        elif pipe.geometry.crosses(vsplitter):
            splitter = vsplitter
        else:
            raise ValueError("point is not on the pipe")
        split_results = complex_split(pipe.geometry, splitter)
        assert len(split_results.geoms) == 2
        p1, p2 = split_results.geoms
        new_pipe = copy.copy(pipe)
        pipe.geometry = p1
        new_pipe.geometry = p2
        new_node = self.get_or_create_node(point)
        new_node.compid = getattr(pipe, 'compid', None)
        pipe.node2 = new_node
        new_pipe.node1 = new_node
        new_pipe.id = self.find_unique_pipe_id(pipe.id)
        self.add_pipe(new_pipe)
        return pipe, new_pipe, new_node

    def add_pipe(self, pipe):
        """
        Add a new pipe to the pipe dict, and also update all related data
        structures.
        """
        self.pipes[pipe.id] = pipe
        if pipe.type == 'main':  # add reference to trunkmain GIS
            entry = self._pipe_map.setdefault(pipe.ref, [])
            entry.append(pipe)
        pipe.node1.pipes.append(pipe)
        pipe.node2.pipes.append(pipe)

    def find_dma(self, geometry):
        candidates = list(self.gis_dma.sindex.intersection(geometry.bounds))
        for c in candidates:
            dma = self.gis_dma.iloc[c]
            if dma.geometry.contains(geometry):
                return dma
        return None

    def find_closest_pipe(self, point, start_dist=50, force_main_comp=True, excluded=[]):
        """
        Find the closest trunkmain pipe to the specified point, excluding
        zero-length pipes representing valves on trunkmain
        """
        dist = start_dist
        while True:
            result_orig = self.gis_trunkmain.sindex.intersection(point.buffer(dist).bounds)
            result_orig = list(result_orig)

            # each of the resulting original pipes may contain more than one
            # derived pipes; combine them all into a single list
            nearby_pipes = sum([self._pipe_map[r] for r in result_orig], [])
            nearby_pipes = [p for p in nearby_pipes if p not in excluded]

            if force_main_comp:
                # consider only 'connected' pipes and ignore 'valve' pipes
                nearby_pipes = [p for p in nearby_pipes
                                  if getattr(p, "compid", None) == 1
                                     and not p.type.endswith('_valve')]
            else:
                # ignore 'valve' pipes
                nearby_pipes = [p for p in nearby_pipes
                                  if not p.type.endswith('_valve')]

            if nearby_pipes:
                break
            dist *= 2
            if dist > 1e6:
                return None

        dists = np.array([p.geometry.distance(point) for p in nearby_pipes])
        idx = dists.argmin()
        return nearby_pipes[idx]

    def get_or_create_attach_point(self, point, pipe=None, max_dist=None):
        """
        Find the closest pipe, if not given, to attach to the given point.  If
        the given point is located on one of the pipe's endpoints, the
        endpoint is used.  Otherwise, the pipe is split with a new node
        created as the attach point.  Return the endpoint and the associated
        pipe, along with all the pipes attached to the same endpoint.
        """

        if pipe is None:
            pipe = self.find_closest_pipe(point)

        if pipe is None:
            return None, None, None

        # find the closest point along the pipe
        point_on_pipe = nearest_points(pipe.geometry, point)[0]

        # find the closer endpoint of the pipe
        head, tail = pipe.geometry.boundary.geoms
        dist_to_head = point_on_pipe.distance(head)
        dist_to_tail = point_on_pipe.distance(tail)
        if dist_to_head <= dist_to_tail:
            closer_node = pipe.node1
            min_dist = dist_to_head
        else:
            closer_node = pipe.node2
            min_dist = dist_to_tail

        if max_dist is not None and point_on_pipe.distance(point) > max_dist:
            # cannot find a pipe closer than the specified distance
            return None, None, None

        # if the attach point is very close to the closer node, just use the
        # node as the attach point.  Otherwise, split the pipe into two
        # subpipes.
        if min_dist < SNAP_THRESHOLD:
            # also return the other pipes attached to the same node
            attached_pipes = set(p.id for p in closer_node.pipes)
            other_pipes = attached_pipes - {pipe.id}
            return pipe, closer_node, [self.pipes[id] for id in other_pipes]
        else:
            logger.debug(f"Pipe {pipe.id} split into two")
            pipe, new_pipe, new_node = self.split_pipe(pipe, point_on_pipe)
            return pipe, new_node, [new_pipe]

    def process_pipes(self):
        # TODO check invalid diameter, etc.
        for i, pipe_row in self.gis_trunkmain.iterrows():
            try:
                pipe_char = self.df_pipechar.loc[pipe_row.matl, pipe_row.pipe_size]
            except KeyError:
                logger.warning(f"No characteristics found for pipe at row {i+1}; "
                         f"matl={pipe_row.matl}, size={pipe_row.pipe_size}")
                pipe_char = None
            geom = pipe_row.geometry.coords
            head = self.get_or_create_node(Point(*geom[0]))
            tail = self.get_or_create_node(Point(*geom[-1]))
            if pipe_char is not None:
                diameter = pipe_char.ext_size - 2*pipe_char.thick
                # TODO what if inst_year is None
                try:
                    inst_year = int(pipe_row.inst_year)
                except TypeError:
                    inst_year = 50
                # apply deprecation (ROUGHNESS_RATE/year), no more than 50 years
                roughness = pipe_char.rough - ROUGHNESS_RATE*min(50, YEAR - inst_year)
            else:
                diameter = pipe_row.pipe_size
                roughness = 999
            if diameter == 0:
                logger.warning(f"Pipe {i} has zero diameter; set to default of {DEFAULT_PIPE_DIAMETER_SIZE}")
                diameter = DEFAULT_PIPE_DIAMETER_SIZE
            pipe = Pipe(id=str(i),
                        type="main",
                        node1=head,
                        node2=tail,
                        diameter=diameter,
                        roughness=roughness,
                        minorloss=0,
                        status="OPEN",
                        row=str(i+1),
                        comment=f"{pipe_row.matl}{pipe_row.inst_year}",
                        geometry=pipe_row.geometry,
                        )
            # check if the pipe's linestring is self-crossed
            if not pipe_row.geometry.is_simple:
                logger.warning(f"Pipe {i} is self-crossed")
            pipe.ref = i  # reference to the zero-based index of the original dataframe
            self.add_pipe(pipe)

    def snap_pipes(self):
        """
        Split a pipe laying on another pipe's endpoint and snap them all
        """
        while True:
            split_count = 0
            for pipe in list(self.pipes.values()):
                # only consider trunkmain pipe
                if pipe.type != "main":
                    continue
                # ignore all short pipes
                if pipe.geometry.length < SNAP_PIPE_LENGTH_THRESHOLD:
                    continue
                for node in [pipe.node1, pipe.node2]:
                    if len(node.pipes) > 1: # ignore non-dead-end nodes
                        continue
                    closest_pipe = self.find_closest_pipe(node.geometry,
                                                          force_main_comp=False,
                                                          excluded=[pipe])
                    if node.geometry.distance(closest_pipe.geometry) > SNAP_PIPE_MAX_DISTANCE:
                        continue
                    if closest_pipe in node.pipes:
                        continue
                    logger.info(f"Snapping endpoint of pipe {pipe.id} to pipe {closest_pipe.id}")
                    split_count += 1
                    pipe.flag = 1
                    closest_pipe.flag = 1
                    xpipe, xnode, _  = self.get_or_create_attach_point(node.geometry,
                                                                       pipe=closest_pipe,
                                                                       max_dist=SNAP_THRESHOLD)
                    assert xpipe is closest_pipe
                    if xnode is not node:
                        # swap out the pipe's endpoint and replace it with xnode
                        assert len(node.pipes) == 1
                        assert pipe not in xnode.pipes
                        xnode.pipes.append(pipe)
                        pipe_coords = pipe.geometry.coords[:]
                        if node is pipe.node1:
                            assert pipe.geometry.coords[0] == node.geometry.coords[0]
                            pipe.node1 = xnode
                            pipe_coords[0] = xnode.geometry.coords[0]
                        elif node is pipe.node2:
                            assert pipe.geometry.coords[-1] == node.geometry.coords[0]
                            pipe.node2 = xnode
                            pipe_coords[-1] = xnode.geometry.coords[0]
                        else:
                            assert False
                        pipe.geometry = LineString(pipe_coords)
                        node.pipes.remove(pipe)
            if split_count == 0:  # no more splitting detected
                break

    def process_valve_on_main(self):
        # TODO ค่า K ทำเป็น 0 ไว้ก่อน ภายหลัง ให้ใช้คอลัมน์ STATUS_K
        # TODO เอาคอลัมน์ STATUS มาใส่เป็น ; COMMENT ไม่ได้เนื่องจากไม่มีข้อมูล GIS จริง ๆ
        count = 1
        for vindex, vrow in self.gis_valve.iterrows():
            vname = f'VAL-{count}'
            pipe, vnode1, other_pipes = self.get_or_create_attach_point(vrow.geometry)
            dist_to_pipe = vrow.geometry.distance(pipe.geometry)
            if dist_to_pipe > VALVE_ON_MAIN_THRESHOLD:
                logger.info(f"Skipping valve {vrow.vid} because it's too far from {pipe.id} ({dist_to_pipe}m)")
                continue
            if len(other_pipes) == 0:
                logger.info(f"Skipping valve {vrow.vid} because it's a deadend")
                continue
            if len(other_pipes) > 1 :
                logger.info(f"Skipping valve {vrow.vid} because it connects more than two pipes")
                continue
            #print(dist_to_pipe, len(other_pipes))
            #assert len(other_pipes) == 1

            pipe1 = pipe
            pipe2 = other_pipes[0]
            logger.info(f"Creating valve {vrow.vid} between {pipe1.id} and {pipe2.id} as {vname}")

            # create a zero-length pipe representing the valve
            #  |-------------||---------------|
            #        pipe   pipe_v  new_pipe
            self.update_node_id(vnode1, f'{vname}-1', avoid_duplicate=False)

            # clone the node at the same point and connect them with a dummy
            # zero-length pipe
            vnode2 = self.get_or_create_node(vnode1.geometry, always_create=True)
            vnode2.compid = vnode1.compid
            self.update_node_id(vnode2, f'{vname}-2', avoid_duplicate=False)

            # create the dummy zero-length pipe with all the same pipe's properties
            # and insert it between the two valve-nodes
            vpipe = copy.copy(pipe)
            vpipe.id = vname
            vpipe.type = 'main_valve'
            vpipe.node1 = vnode1
            vpipe.node2 = vnode2
            vpipe.geometry = LineString([vnode1.geometry, vnode2.geometry])
            vpipe.valve_type = "TCV"
            vpipe.setting = 0
            if vrow.percent is not None:
                try:
                    vpipe.setting = 1580.7*math.exp(-0.090309*float(vrow.percent))
                except ValueError:
                    logger.warning(f"Invalid percent value for valve on main '{vname}'")
            else:
                logger.warning(f"Valve on main '{vname}' has no percent value")
            self.add_pipe(vpipe)

            # we need to find out which endpoints of the corredponding pipes this valve-node is
            # attached to; then reconnect one of the pipes thru the valve-nodes
            if pipe2.node1 == vnode1:
                pipe2.node1 = vnode2
            else:
                pipe2.node2 = vnode2

            count += 1

    def process_uuz(self):
        pipe_list = list(self.pipes.values())
        for uuz_name, uuz_row in self.gis_uuz.iterrows():
            # find point on the pipe to attach the U/UZ; split pipe if needed
            pipe, uuz_node, _ = self.get_or_create_attach_point(uuz_row.geometry)

            if pipe is None:
                logger.warning(f"Cannot find attach point for {uuz_name}")
                continue

            # associate the U/UZ to the pipe's endpoint
            uuz_name = self.update_node_id(uuz_node, uuz_name, avoid_duplicate=True)
            uuz_node.type = "uuz"
            logger.info(f"Associate {uuz_name} to pipe {pipe.id} (row={pipe.row})")

            # rename the pipe to the UUZ name
            self.update_pipe_id(pipe, uuz_name, avoid_duplicate=True)

            # record the distance from the original location
            uuz_node.reloc_dist = uuz_row.geometry.distance(uuz_node.geometry)

    def process_dm(self):
        patterns = set([name[:-1] for name in self.df_f_pattern.index])
        for dm_name, dm in self.gis_dm.iterrows():
            if len(dm_name) != 8:
                logger.warning(f'Invalid DM name {dm_name}')
                continue
            dm_name = f'DM-{dm_name[0:2]}-{dm_name[2:4]}-{dm_name[4:6]}-{dm_name[6:8]}'
            if dm_name in self.nodes:
                logger.warning(f'{dm_name} already exists')
                continue
            dm_node = self.get_or_create_node(dm.geometry, always_create=True)
            dm_node.type = "dm"
            self.update_node_id(dm_node, dm_name, avoid_duplicate=False)
            if dm_name in patterns:
                dm_node.pattern = f"{dm_name}-F"
            else:
                dm_node.pattern = "DEFAULTFLOW-F"

            try:
                dm_node.demand = self.df_f_avg.loc[dm_name + 'F'].base
            except KeyError:
                logger.warning(f'{dm_name} not found in default flow table')
                dm_node.demand = 0

            # find the closest pipe to attach the DM; split pipe if needed
            pipe, attach_node, _ = self.get_or_create_attach_point(dm.geometry, max_dist=MAX_DM_DISTANCE)

            if pipe is None:
                logger.warning(f"Cannot find attach point for {dm_name}")
                continue

            dm_node.compid = pipe.compid
            dist = dm_node.geometry.distance(attach_node.geometry)
            logger.info(f"Link DM {dm_name} to pipe {pipe.id} (row={pipe.row},dist={dist:.3f})")

            # create a valve from DM to the split point
            valve = Pipe(id=f"TCV-{dm_name}",
                         node1=attach_node,
                         node2=dm_node,
                         type="dm_valve",
                         diameter=dm.pipe_size or DEFAULT_DM_VALVE_SIZE,
                         roughness=0,  # not use
                         minorloss=0,  # not use
                         row=0,        # not use
                         status='',    # not use
                         comment='',
                         geometry=LineString([attach_node.geometry, dm_node.geometry]))
            valve.setting = 0  # XXX
            valve.valve_type = "TCV"
            valve.compid = getattr(pipe, 'compid', None)
            self.add_pipe(valve)

    def process_resv(self):
        patterns = set(self.df_resv_pattern.index)
        for resv_name, resv in self.gis_resv.iterrows():
            try:
                type_entry = self.df_resv_type.loc[resv_name]
            except KeyError:
                logger.warning(f"Cannot find information for reservoir {resv_name}")
                continue
            resv_node = self.get_or_create_node(resv.geometry, always_create=True)
            resv_node.type = "resv"
            resv_node.resv_type = type_entry.type
            if resv_node.resv_type == 'NODE':
                resv_node.demand = -type_entry.demand_or_head
                resv_node.pattern = f"{resv_name}-F"
            else:
                resv_node.head = type_entry.demand_or_head
                resv_node.pattern = f"{resv_name}-P"
            self.update_node_id(resv_node, resv_name, avoid_duplicate=False)

            # find the closest pipe to attach the reservoir node; split pipe if needed
            pipe, attach_node, _ = self.get_or_create_attach_point(resv.geometry)

            if pipe is None:
                logger.warning(f"Cannot find attach point for {resv_name}")
                continue

            resv_node.compid = pipe.compid
            logger.info(f"Link {resv_name} to pipe {pipe.id} (row={pipe.row})")

            # create a valve from RESV to the split point
            valve = Pipe(id=f"{resv_name}",
                         node1=attach_node,
                         node2=resv_node,
                         type="resv_valve",
                         diameter=DEFAULT_RESV_VALVE_SIZE,
                         roughness=0,  # not use
                         row=0,        # not use
                         minorloss=0,  # not use
                         status='',    # not use
                         comment='',
                         geometry=LineString([attach_node.geometry, resv_node.geometry]))
            valve.valve_type = "TCV"
            valve.setting = 0  # XXX
            valve.compid = getattr(pipe, 'compid', None)
            self.add_pipe(valve)

    def resolve_junctions_and_vertices(self):
        logger.info("Resolving junctions and vertices")
        self.junctions = []
        self.vertices = []
        for node in self.nodes.values():
            # Reservoir of type RES will not show up in [JUNCTIONS]
            if node.type == 'resv' and node.resv_type == 'RES':
                continue
            if getattr(node, 'compid', None) != 1:
                logger.warning(f"Node {node.id} is disconnected from the main group")
                continue
            demand = getattr(node, 'demand', 0)
            pattern = getattr(node, 'pattern', '')
            dma = self.find_dma(node.geometry)
            blockname = dma.blockname if dma is not None else ''
            self.junctions.append(
                Junction(node=node,
                         elev=0,
                         demand=demand,
                         pattern=pattern,
                         comment=blockname))

        for pipe in self.pipes.values():
            if getattr(pipe, 'compid', None) != 1:
                logger.warning(f"Pipe {pipe.id} is disconnected from the main group")
                continue
            for x,y in pipe.geometry.coords[1:-1]:
                # exclude coordinates of pipe's endpoints
                self.vertices.append(Vertex(pipe, x, y))

    def generate_title(self):
        logger.info("Generating [TITLE]")
        self.output("[TITLE]")
        self.output("Metropolitan Waterworks Authority Trunk Main Network Model")

    def generate_junctions(self):
        logger.info("Generating [JUNCTIONS]")
        self.output("\n[JUNCTIONS]")
        self.output(f";{'ID':<18} {'Elev':<15} {'Demand':<15} {'Pattern':<18}")
        for junc in sorted(self.junctions, key=lambda j:j.node.id):
            self.output(f" {junc.node.id:<18} {junc.elev:<15} {junc.demand:<15.4f} {junc.pattern:<18} ;{junc.comment}")

    def generate_reserviors(self):
        logger.info("Generating [RESERVOIRS]")
        self.output("\n[RESERVOIRS]")
        self.output(";"
                f"{'ID':<18} "
                f" {'Head':<15} "
                f" {'Pattern':<19}")
        for resv_name, resv in self.df_resv_type.iterrows():
            if resv.type == 'NODE':
                continue
            resv_node = self.nodes[resv_name]
            self.output(" "
                    f"{resv_name:<18} "
                    f" {resv_node.head:<15} "
                    f" {resv_node.pattern:<19}")

    def generate_tanks(self):
        pass

    def generate_pipes(self):
        logger.info("Generating [PIPES]")
        self.output("\n[PIPES]")
        self.output(";"
                f"{'ID':<18} "
                f"{'Node1':<19} "
                f"{'Node2':<19} "
                f"{'Length':<15} "
                f"{'Diameter':<15} "
                f"{'Roughness':<15} "
                f"{'MinorLoss':<15} "
                f"{'Status'} ")
        for id in sorted(self.pipes):
            pipe = self.pipes[id]
            if pipe.type != 'main':
                continue
            self.output(" "
                    f"{pipe.id:<18} "
                    f"{pipe.node1.id:<19} "
                    f"{pipe.node2.id:<19} "
                    f"{pipe.geometry.length:<15.5f} "
                    f"{pipe.diameter:<15.2f} "
                    f"{pipe.roughness:<15} "
                    f"{pipe.minorloss:<15} "
                    f"{pipe.status:8} " 
                    f";{pipe.comment}")

    def generate_pumps(self):
        pass

    def generate_valves(self):
        logger.info("Generating [VALVES]")
        valves = [v for v in self.pipes.values() 
                    if v.type in ['dm_valve', 'main_valve', 'resv_valve']]
        self.output("\n[VALVES]")
        self.output(";"
                f"{'ID':<24} "
                f"{'Node1':<20} "
                f"{'Node2':<20} "
                f"{'Diameter':<16} "
                f"{'Type':<8} "
                f"{'Setting':<16} "
                f"{'Minorloss':<16} "
                )
        for v in valves:
            self.output(" "
                    f"{v.id:<24} "
                    f"{v.node1.id:<20} "
                    f"{v.node2.id:<20} "
                    f"{v.diameter:<16.2f} "
                    f"{v.valve_type:<8} "
                    f"{v.setting:<16.6f} "
                    f"{v.minorloss:<16} "
                    f";{v.comment}"
                    )

    def generate_tags(self):
        pass

    def generate_demands(self):
        pass

    def generate_status(self):
        pass

    def generate_patterns(self):
        logger.info("Generating [PATTERNS]")
        self.output("\n[PATTERNS]")
        self.output(";"
                f"{'ID':<18} "
                f"Multipliers")
        cols = [f'h{h}' for h in range(24)]

        for name, pattern in self.df_resv_pattern.iterrows():
            if pd.isna(pattern[cols]).any():  # some values are missing
                logger.warning(f"Some pattern values of {name} are missing")
            self.output(f";{name}")
            for c in range(0, 24, 6):
                vals = pattern[cols[c:c+6]]
                self.output(" {:18} {}".format(
                    f'{name}-{pattern.type}',
                    ' '.join(f"{v:11.9f}" for v in vals),
                    )
                )

        for name, pattern in self.df_f_pattern.iterrows():
            if pd.isna(pattern[cols]).any():  # some values are missing
                logger.warning(f"Some flow pattern values of {name} are missing")
            if name != "DEFAULTFLOW":
                name = name[:-1]
            self.output(f";{name}")
            for c in range(0, 24, 6):
                vals = pattern[cols[c:c+6]]
                self.output(" {:18} {}".format(
                    f'{name}-F',
                    ' '.join(f"{v:11.9f}" for v in vals),
                    )
                )

        for name, pattern in self.df_p_pattern.iterrows():
            if pd.isna(pattern[cols]).any():  # some values are missing
                logger.warning(f"Some pressure pattern values of {name} are missing")
            if name != "DEFAULTFLOW":
                name = name[:-1]
            self.output(f";{name}")
            for c in range(0, 24, 6):
                vals = pattern[cols[c:c+6]]
                self.output(" {:18} {}".format(
                    f'{name}-P',
                    ' '.join(f"{v:11.9f}" for v in vals),
                    )
                )

    def generate_curves(self):
        pass

    #def generate_controls(self):
    #    self.output("\n[CONTROLS]")
    #    self.output(self.txt_prv)

    def generate_rules(self):
        pass

    def generate_energy(self):
        pass

    def generate_emitters(self):
        # TODO DM don't have emitter (only items on trunkmain)
        # NOTES will be processed later by Python-EPANET
        pass

    def generate_quality(self):
        pass

    def generate_sources(self):
        pass

    def generate_reactions(self):
        pass

    def generate_mixing(self):
        pass

    def generate_times(self):
        pass

    def generate_report(self):
        pass

    def generate_options(self):
        pass

    def generate_coordinates(self):
        logger.info("Generating [COORDINATES]")
        self.output("\n[COORDINATES]")
        self.output(f";{'Node':<16} {'X-Coord':<16} {'Y-Coord':<16}")
        for id in sorted(self.nodes):
            node = self.nodes[id]
            x, y = node.geometry.coords[0]
            self.output(f" {node.id:<16} {x:<16.2f} {y:<16.2f}")

    def generate_vertices(self):
        logger.info("Generating [VERTICES]")
        self.output("\n[VERTICES]")
        self.output(f";{'Link':<16} {'X-Coord':<16} {'Y-Coord':<16}")
        for v in sorted(self.vertices, key=lambda v: v.pipe.id):
            self.output(f" {v.pipe.id:<16} {v.x:<16.2f} {v.y:<16.2f}")

    def generate_labels(self):
        pass

    def generate_backdrop(self):
        pass

    def generate_output(self):
        self.generate_title()
        self.generate_junctions()
        self.generate_reserviors()
        self.generate_tanks()
        self.generate_pipes()
        self.generate_pumps()
        self.generate_valves()
        self.generate_tags()
        self.generate_demands()
        self.generate_status()
        self.generate_patterns()
        self.generate_curves()
        #self.generate_controls()
        self.generate_rules()
        self.generate_energy()
        self.generate_emitters()
        self.generate_quality()
        self.generate_sources()
        self.generate_reactions()
        self.generate_mixing()
        self.generate_times()
        self.generate_report()
        self.generate_options()
        self.generate_coordinates()
        self.generate_vertices()
        self.generate_labels()
        self.generate_backdrop()

    def calculate_connectivity(self, start_pipe):
        """Perform depth-first search to check connectivity"""
        unexplored = set(self.pipes)
        discovered = set()
        compid = 0
        visiting = [start_pipe]
        while unexplored:
            compid += 1
            if not visiting:
                visiting = [next(iter(unexplored))]
            while visiting:
                current = self.pipes[visiting.pop()]
                current.compid = compid
                current.node1.compid = compid
                current.node2.compid = compid
                if current.id in unexplored:
                    unexplored.remove(current.id)
                discovered.add(current.id)
                for pipe in current.node1.pipes + current.node2.pipes:
                    if pipe.id not in discovered:
                        visiting.append(pipe.id)

        # remap component id to sort connected components by their sizes
        compsize = {}
        for p in self.pipes.values():
            compsize[p.compid] = compsize.get(p.compid, 0) + 1
        size_pairs = list(compsize.items())
        size_pairs.sort(key=lambda x: (x[1], x[0]), reverse=True)
        remap = {oldid:newid for newid,(oldid,size) in enumerate(size_pairs, 1)}
        for p in self.pipes.values():
            p.compid = remap[p.compid]
        for n in self.nodes.values():
            if hasattr(n, "compid"):
                n.compid = remap[n.compid]
            else:
                n.compid = None

    def save(self, outfile):
        # copy certain data and get rid of references to avoid recursive pickling
        nodes_copy = copy.copy(self.nodes)
        for n in nodes_copy.values():
            n.pipes = [p.id for p in n.pipes]

        pipes_copy = copy.copy(self.pipes)
        for p in pipes_copy.values():
            p.node1 = p.node1.id
            p.node2 = p.node2.id

        junctions_copy = copy.copy(self.junctions)
        for j in junctions_copy:
            j.node = j.node.id

        vertices_copy = copy.copy(self.vertices)
        for v in vertices_copy:
            v.pipe = v.pipe.id

        data = {
                'nodes': nodes_copy,
                'pipes': pipes_copy,
                'junctions': junctions_copy,
                'vertices': vertices_copy,
                '_node_counter': self._node_counter,
        }
        for a in ATTR_MAPPING.values():
            data[a] = getattr(self, a)

        with open(outfile, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Current data saved to {outfile}")

    def load(self, infile):
        with open(infile, "rb") as f:
            data = pickle.load(f)

        for a in ATTR_MAPPING.values():
            setattr(self, a, data[a])

        # for some reason, sindex is corrupted when loaded from pickle, so we
        # need to create a copy to force removal of sindex
        self.gis_trunkmain = self.gis_trunkmain.copy()
        self.pipes = data['pipes']
        self.nodes = data['nodes']
        self.junctions = data['junctions']
        self.vertices = data['vertices']
        self._node_counter = data['_node_counter']

        # resolve original references from object ids
        self._node_map_by_coords = {}
        self._pipe_map = {}
        for p in self.pipes.values():
            p.node1 = self.nodes[p.node1]
            p.node2 = self.nodes[p.node2]
            if hasattr(p, 'ref'):
                pipe_map_entry = self._pipe_map.setdefault(p.ref, [])
                pipe_map_entry.append(p)
        for n in self.nodes.values():
            n.pipes = [self.pipes[p] for p in n.pipes]
            coords_key = self.get_coords_map_key(n.geometry)
            self._node_map_by_coords[coords_key] = n
        for j in self.junctions:
            j.node = self.nodes[j.node]
        for v in self.vertices:
            v.pipe = self.pipes[v.pipe]

        logger.info(f"Loaded data from {infile}")

    def export_pipes(self, file):
        logger.info("Exporting GIS for pipes")
        fields = {
            'name': [p.id for p in self.pipes.values()],
            'row': [p.row for p in self.pipes.values()],
            'type': [p.type for p in self.pipes.values()],
            'flag': [p.flag for p in self.pipes.values()],
            'compid': [getattr(p, 'compid', None) for p in self.pipes.values()],
        }
        df = gpd.GeoDataFrame(
                index=list(self.pipes.keys()),
                data=fields,
                geometry=[p.geometry for p in self.pipes.values()],
                crs="EPSG:32647")
        df.to_crs("EPSG:4326").to_file(file)

    def export_nodes(self, file):
        logger.info("Exporting GIS for nodes")
        df = gpd.GeoDataFrame(
                index=list(self.nodes.keys()),
                data={
                    'type': [n.type for n in self.nodes.values()],
                    'compid': [getattr(n, 'compid', None) for n in self.nodes.values()],
                },
                geometry=[n.geometry for n in self.nodes.values()],
                crs="EPSG:32647")
        df.to_crs("EPSG:4326").to_file(file)

    def export_uuz(self, file):
        logger.info("Exporting GIS for U/UZ")
        uuz = [n for n in self.nodes.values() if n.type == "uuz"]
        df = gpd.GeoDataFrame(
                index=[n.id for n in uuz],
                data={
                    'reloc_dist': [n.reloc_dist for n in uuz],
                },
                geometry=[n.geometry for n in uuz],
                crs="EPSG:32647")
        df.to_crs("EPSG:4326").to_file(file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} step".format(sys.argv[0]))
        sys.exit(1)

    step = int(sys.argv[1])
    if step == 0:
        epanet = Epanet()
        epanet.read_inputs()
        epanet.save("step0.pickle")
    elif step == 1:
        epanet = Epanet("step0.pickle")
        epanet.process_pipes()
        # epanet.export_pipes("check/pipes-1.shp")
        epanet.save("step1.pickle")
    elif step == 2:
        epanet = Epanet("step1.pickle")
        epanet.snap_pipes()
        # epanet.export_pipes("check/pipes-2.shp")
        epanet.save("step2.pickle")
    elif step == 3:
        epanet = Epanet("step2.pickle")
        epanet.calculate_connectivity(start_pipe='1348') # TODO avoid hardcode
        epanet.process_uuz()
        epanet.process_valve_on_main()
        epanet.save("step3.pickle")
    elif step == 4:
        epanet = Epanet("step3.pickle")
        epanet.process_dm()
        epanet.process_resv()
        os.makedirs("check", exist_ok=True)
        epanet.export_pipes("check/pipes.shp")
        epanet.export_nodes("check/nodes.shp")
        epanet.save("step4.pickle")
    elif step == 5:
        epanet = Epanet("step4.pickle")
        epanet.resolve_junctions_and_vertices()
        with open("epanet.inp", "w", encoding="utf-8") as f:
            epanet.output_stream = f
            epanet.generate_output()
