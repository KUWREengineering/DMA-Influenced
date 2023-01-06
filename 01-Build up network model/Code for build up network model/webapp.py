from flask import Flask, render_template, url_for, request
from epanet_prep import (
    Epanet,
    Node,
    Pipe,
    Junction,
    Vertex,
    Emitter,
)
import geopandas as gpd
from shapely.geometry import Point

QUERIES = {
    'all': {
        'desc': 'All pipes and nodes',
        'func': 'query_all',
    },
    'disconnect': {
        'desc': 'Disconnected pipes',
        'func': 'query_disconnect',
    },
    'branch': {
        'desc': 'Pipe branching',
        'func': 'query_branch',
    },
}

SRS = "EPSG:32647"

app = Flask(__name__)
epanet = Epanet('step2.pickle')

def gpd_for_pipes(pipe_list):
    refs = [int(p.ref) for p in pipe_list]
    df = gpd.GeoDataFrame(
            index=[p.id for p in pipe_list],
            data={
                'ref': [int(p.ref) for p in pipe_list],
                'highlight': False,
            },
            geometry=[p.geometry for p in pipe_list],
            crs=SRS)
    return df.to_crs("EPSG:4326")

def gpd_for_nodes(node_list):
    df = gpd.GeoDataFrame(
            index=[n.id for n in node_list],
            data={
                'highlight': False,
            },
            geometry=[n.geometry for n in node_list],
            crs=SRS)
    return df.to_crs("EPSG:4326")

def get_local_pipes_and_nodes(pipe):
    local_pipes_set = set([pipe.id])
    local_pipes_set.update(set([p.id for p in pipe.node1.pipes]))
    local_pipes_set.update(set([p.id for p in pipe.node2.pipes]))
    local_pipes = [epanet.pipes[id] for id in local_pipes_set]

    local_nodes_set = set()
    for p in local_pipes:
        local_nodes_set.add(p.node1.id)
        local_nodes_set.add(p.node2.id)
    local_nodes = [epanet.nodes[id] for id in local_nodes_set]

    return local_pipes, local_nodes

def query_all():
    nodes = list(epanet.nodes.values())
    pipes = list(epanet.pipes.values())
    return nodes, pipes

def query_disconnect():
    nodes = [n for n in epanet.nodes.values() if len(n.pipes) == 1]
    pipes_set = set()
    for n in nodes:
        pipes_set.update([p.id for p in n.pipes])
    pipes = [epanet.pipes[p] for p in pipes_set]
    return nodes, pipes

def query_branch():
    nodes = [n for n in epanet.nodes.values() if len(n.pipes) > 2]
    pipes_set = set()
    for n in nodes:
        pipes_set.update([p.id for p in n.pipes])
    pipes = [epanet.pipes[p] for p in pipes_set]
    return nodes, pipes

def get_nodes_and_pipes_for_map(name):
    try:
        query = QUERIES[name]
        func = eval(query['func'])
        nodes, pipes = func()
    except KeyError:
        nodes = []
        pipes = []
    return nodes, pipes

@app.route('/')
def home():
    return render_template('home.html', queries=QUERIES)

@app.route('/pipelist/<name>')
def pipe_list(name):
    nodes, pipes = get_nodes_and_pipes_for_map(name)
    offset = int(request.args.get('offset', 0))
    length = int(request.args.get('length', 50))
    return render_template('pipes.html',
            pipes=pipes[offset:offset+length])

@app.route('/nodelist/<name>')
def node_list(name):
    nodes, pipes = get_nodes_and_pipes_for_map(name)
    offset = int(request.args.get('offset', 0))
    length = int(request.args.get('length', 50))
    return render_template('nodes.html',
            nodes=nodes[offset:offset+length])

@app.route('/map/<name>')
def view_map(name):
    nodes, pipes = get_nodes_and_pipes_for_map(name)
    gpd_nodes = gpd_for_nodes(nodes)
    gpd_pipes = gpd_for_pipes(pipes)
    return render_template('map.html',
            map_name=name,
            node_geom=gpd_nodes,
            pipe_geom=gpd_pipes,
            )

@app.route('/pipe/<id>')
def pipe_details(id):
    pipe = epanet.pipes[id]
    pipe_series = epanet.df_trunkmain.loc[int(pipe.ref)]
    local_pipes, local_nodes = get_local_pipes_and_nodes(pipe)
    gpd_pipes = gpd_for_pipes(local_pipes)
    gpd_nodes = gpd_for_nodes(local_nodes)
    gpd_pipes.loc[id, 'highlight'] = True

    return render_template('pipe-details.html',
            pipe=pipe,
            pipe_series=pipe_series,
            pipe_geom=gpd_pipes,
            node_geom=gpd_nodes,
            )

@app.route('/node/<id>')
def node_details(id):
    node = epanet.nodes[id]
    gpd_nodes = gpd_for_nodes([node])
    gpd_pipes = gpd_for_pipes(node.pipes)
    return render_template('node-details.html',
            node=node,
            node_geom=gpd_nodes,
            pipe_geom=gpd_pipes,
            )

if __name__ == '__main__':
    app.run(debug=True)
    #nodes, pipes = get_nodes_and_pipes_for_map("disconnect")
    #df = gpd_for_pipes(pipes)
    #df.to_file("check/disconnect.shp")

