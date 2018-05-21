from scpd.graph.graph import CsvGraphParser
from scpd.graph.astree import to_ast, GraphvizVisitor
from scpd.graph.visitor import Visitor

if __name__ == "__main__":
    with open("test/parsed/dummy.cpp/nodes.csv") as fn:
        with open("test/parsed/dummy.cpp/edges.csv") as fe:
            parser = CsvGraphParser(fn, fe, id_field="key")
            root = to_ast(parser.parse().node_by_index(0))
            visitor = GraphvizVisitor()
            visitor.visit(root)
            visitor.graph().render('test.gv', view=True)
