from scpd.joern.parser import extract_joern_code
from scpd.source import SourceCode

if __name__ == "__main__":
    with open("test/dummy.cpp") as f:
        source = SourceCode("roberio", f.read())
        print(extract_joern_code(source, cache_along=False))
