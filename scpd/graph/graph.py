from copy import deepcopy
from .. import csv


class Edge():
    def __init__(self, start, end, kind=None, row=None):
        if not isinstance(start, Node):
            raise AssertionError("start should be a Node")
        if not isinstance(end, Node):
            raise AssertionError("end should be a Node")
        self._a = start
        self._b = end
        self._kind = kind
        self._row = row
        self._reversed = False

        if kind is None and row is not None and row.has("type"):
            self._kind = row.get("type")

    def start(self):
        return self._a

    def end(self):
        return self._b

    def type(self):
        if self._reversed:
            return None
        return self._kind

    def reverse(self):
        self._a, self._b = self._b, self._a
        self._reversed = not self._reversed
        self._kind = None

    def clone(self):
        return deepcopy(self)


class Node():
    def __init__(self, identifier=None, row=None):
        if identifier is None:
            if row is None:
                raise AssertionError("CSV row should be given")
            if not isinstance(row, csv.NamedRow):
                raise AssertionError("CSV row should be a NamedRow")
            if not row.has("id"):
                raise AssertionError("row should have id field")
            self._id = row.get("id")
        else:
            self._id = identifier
        self._row = row
        self._edges = []

    def id(self):
        return self._id

    def add_edge(self, edge):
        self._edges.append(edge)

    def add_edges(self, edges):
        self._edges.extend(edges)

    def edges(self):
        return self._edges

    def clone(self):
        return deepcopy(self)


class Graph():
    def __init__(self, nodes=[], edges=None):
        self._nodes = nodes
        self._build_node_dict()
        if edges is not None:
            self.add_edges(edges)

    def _build_node_dict(self):
        self._node_dict = {}
        for node in self._nodes:
            self._node_dict[node.id()] = node

    def add_node(self, node):
        if node.id() in self._node_dict:
            raise AssertionError("every node should have unique id in a graph")
        self._node_dict[node.id()] = node
        self._nodes.append(node)

    def has_node(self, i):
        if isinstance(i, Node):
            i = i.id()
        return i in self._node_dict

    def node(self, i):
        if isinstance(i, Node):
            i = i.id()
        return self._node_dict[i]

    def node_by_index(self, i):
        return self._nodes[i]

    def add_edge(self, edge):
        start = edge.start()
        end = edge.end()
        if not self.has_node(start):
            raise AssertionError(
                "start node from an edge should have valid id")
        if not self.has_node(end):
            raise AssertionError("end node from an edge should have valid id")
        start.add_edge(edge)

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)


class CsvGraphParser():
    def __init__(self,
                 f_nodes,
                 f_edges,
                 id_field="id",
                 start_field="start",
                 end_field="end",
                 type_field="type"):
        self._node_parser = csv.CsvParser(f=f_nodes, header=True)
        self._edge_parser = csv.CsvParser(f=f_edges, header=True)
        self._id_field = id_field
        self._start_field = start_field
        self._end_field = end_field
        self._type_field = type_field

    def parse(self):
        node_header, node_rows = self._node_parser.parse({
            self._id_field: "id",
        })
        edge_header, edge_rows = self._edge_parser.parse({
            self._start_field: "start",
            self._end_field: "end",
            self._type_field: "type"
        })
        graph = Graph()

        if "id" not in node_header:
            raise AssertionError("header of node file should have an id field")

        for node_row in node_rows:
            node = Node(row=node_row)
            graph.add_node(node)

        if "start" not in edge_header:
            raise AssertionError(
                "header of edge file should have a start field")
        if "end" not in edge_header:
            raise AssertionError("header of edge file should have a end field")

        for edge_row in edge_rows:
            start = edge_row.get("start")
            end = edge_row.get("end")
            edge = Edge(graph.node(start), graph.node(end), row=edge_row)
            graph.add_edge(edge)

        return graph
