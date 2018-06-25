from graphviz import Digraph
from . import graph
from .visitor import Visitable, TreeVisitor

AST_EDGES = "IS_FUNCTION_OF_AST IS_FILE_OF IS_AST_PARENT".split()
LOCATION_CHAR = ":"


class Location:
    def __init__(self, s):
        self.line, _, self.start, self.end = s.split(LOCATION_CHAR)


class TreeNode(graph.Node, Visitable):
    def __init__(self,
                 identifier=None,
                 row=None,
                 kind=None,
                 parent=None,
                 children=[]):
        graph.Node.__init__(self, identifier=identifier, row=row)
        Visitable.__init__(self)
        self._parent = parent
        self._children = list(children)
        self._type = kind if kind is not None else row.get("type")

    def accept(self, visitor):
        return isinstance(visitor, TreeVisitor)

    def set_parent(self, p):
        self._parent = p

    def add_child(self, child):
        self._children.append(child)

    def add_children(self, children):
        self._children.extend(children)

    def children(self):
        return self._children

    def type(self):
        return self._type

    def child_of_type(self, kind):
        for child in self._children:
            if child.type() == kind:
                return child
        return None

    def location(self):
        if not self._row.has("location"):
            return None
        return Location(self._row.get("location"))


class GraphvizVisitor(TreeVisitor):
    def __init__(self):
        self._graph = Digraph()

    def visit(self, element):
        self._graph.node(element.id(), label=element.type())
        for child in element.children():
            self._graph.edge(element.id(), child.id())
            self.step(child)

    def graph(self):
        return self._graph


def to_ast(root, parent=None, seen=None):
    if seen is None:
        seen = {}
    if root.id() in seen:
        raise AssertionError("ast edges induce cycles, not a tree")
    seen[root.id()] = True
    children = []
    for edge in root.edges():
        if edge.type() in AST_EDGES:
            children.append(to_ast(edge.end(), parent=root, seen=seen))

    return TreeNode(
        identifier=root._id, row=root._row, parent=parent, children=children)
