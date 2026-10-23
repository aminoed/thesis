import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

p = Path("AdaSub_poly.py")
tree = ast.parse(p.read_text())
nodes = [
    n for n in tree.body
    if isinstance(
        n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    )
]

ns = {}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), str(p), "exec"),
    ns,
)
ns["DATA"] = ns["BuildStates"]()

funcs = [
    n for n in nodes
    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
]
source_text = ast.dump(
    ast.Module(body=funcs, type_ignores=[]),
    include_attributes=False,
)
print("functions:", hashlib.sha256(source_text.encode()).hexdigest())

for j, name in enumerate(["X", "Y"]):
    data = pd.read_csv(
        f"bounce_ball_{name}_200000.csv",
        float_precision="round_trip",
    )
    X = data[["x0", "x1"]].to_numpy(dtype=float)
    y = data[f"y{j}"].to_numpy(dtype=float)

    for degree in [1, 2, 3]:
        Phi, _ = ns["build_monomial_matrix"](X, degree)
        digest = hashlib.sha256(Phi.astype("<f8").tobytes()).hexdigest()
        del Phi

        w, eps = ns["fit_polynomial_pac"](X, y, degree)
        print(
            "output:", j,
            "degree:", degree,
            "Phi:", digest,
            "eps:", repr(eps),
            "w:", w.tolist(),
            flush=True,
        )