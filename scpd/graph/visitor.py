from abc import ABCMeta, abstractmethod


class Visitable():
    __metaclass__ = ABCMeta

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError()


class Visitor():
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit(self, element):
        raise NotImplementedError()

    def step(self, element):
        if isinstance(element, Visitable) and element.accept(self):
            self.visit(element)


class TreeVisitor(Visitor):
    __metaclass__ = ABCMeta
