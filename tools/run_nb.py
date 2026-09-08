"""Execute a notebook in place so the committed .ipynb carries real outputs.

nbconvert's CLI is awkward about in-place execution on Windows paths, and we want
the traceback printed rather than swallowed, so this drives nbclient directly.
"""
import sys, pathlib, nbformat
from nbclient import NotebookClient

path = pathlib.Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb, timeout=1800, kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
    allow_errors=False,
)
try:
    client.execute()
finally:
    nbformat.write(nb, path)          # keep partial outputs even on failure

n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"executed {path.name}: {n_out}/{n_code} code cells produced output")
