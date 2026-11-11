"""Restore removed NumPy scalar aliases in chumpy 0.70 for NumPy 1.26."""
from importlib.metadata import distribution

package = distribution("chumpy")
if package.version != "0.70":
    raise SystemExit("Patch supports chumpy 0.70 only")
path = package.locate_file("chumpy/__init__.py")
old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
new = "from builtins import bool, int, float, complex, object, str\nunicode = str\nfrom numpy import nan, inf"
text = path.read_text()
if old in text:
    path.write_text(text.replace(old, new))
    print("Applied chumpy NumPy compatibility patch:", path)
elif new in text:
    print("chumpy compatibility patch already applied")
else:
    raise SystemExit("Unexpected chumpy source; refusing to patch")
