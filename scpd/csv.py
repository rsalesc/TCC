import csv


class Row():
    def __init__(self, row):
        self._row = row

    def get(self, i):
        return self._row[i]

    def __str__(self):
        return str(self._row)


class NamedRow(Row):
    def __init__(self, row, names, name_dict=None):
        super().__init__(self)
        if len(row) != len(names):
            raise AssertionError("row length and header length mismatch")
        self._names = names
        if name_dict is None:
            name_dict = {}
            for index, name in enumerate(names):
                name_dict[name] = index
        self._name_dict = name_dict

    def get(self, i):
        if isinstance(i, str):
            i = self._name_dict[i]
        return Row.get(self, i)

    def has(self, i):
        if isinstance(i, str):
            return i in self._name_dict
        return i < len(self._row)


class CsvParser():
    def __init__(self,
                 f=None,
                 text=None,
                 header=False,
                 delimiter='\t',
                 quotechar='"'):
        self._f = f
        self._text = None
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._header = header

    def _read_file(self, f):
        """Returns the header and an iterable for rows."""
        rows = csv.reader(
            self._f, delimiter=self._delimiter, quotechar=self._quotechar)
        if self._header:
            header = next(rows)
            return header, rows
        return [], rows

    def _parse_list(self):
        if self._f is not None:
            return self._read_file(self._f)
        raise NotImplementedError()

    def parse(self, replacers={}):
        header, rows = self._parse_list()
        if self._header:
            header_dict = {}
            for index, name in enumerate(header):
                if name in replacers:
                    name = replacers[name]
                    header[index] = name
                header_dict[name] = index
            if len(header_dict) != len(header):
                raise AssertionError("column names should be unique")
            return header, list(
                map(lambda x: NamedRow(x, header, header_dict), rows))
        else:
            return list(map(lambda x: Row(x), rows))
