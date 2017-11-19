"""Functions for working with strings.
"""

def utf8_str(obj):
    return str(str(obj).encode("utf-8"))
