import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

def run_nb(path, stop=None, skip_save=False):
    nb = json.load(open(path))
    ns = {"__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        if stop is not None and i > stop:
            break
        src = "".join(c["source"])
        if skip_save:
            src = "\n".join(l for l in src.split("\n")
                            if "save_figure" not in l and ".write_image" not in l)
        try:
            exec(compile(src, f"<{os.path.basename(path)}:cell{i}>", "exec"), ns)
        except SystemExit:
            pass
        except Exception as e:
            print(f"!!! ERROR {path} cell {i}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    return True

def _usage() -> str:
    import glob
    nbs = sorted(glob.glob(os.path.join(REPO, "notebooks", "**", "*.ipynb"), recursive=True))
    rel = [os.path.relpath(p, REPO) for p in nbs]
    return ("Использование: python run_verify.py <путь_к_ноутбуку.ipynb>\n"
            "Например: python run_verify.py notebooks/3_evaluations/3_1_core_metrics.ipynb\n"
            "Доступные ноутбуки:\n  " + "\n  ".join(rel))


if len(sys.argv) < 2:
    print(_usage())
    sys.exit(2)

target = sys.argv[1]
if not os.path.isfile(target):
    print(f"Файл не найден: {target}\n")
    print(_usage())
    sys.exit(2)

ok = run_nb(target, skip_save=True)
print(("OK  " if ok else "FAIL") + " " + target)
sys.exit(0 if ok else 1)
