import subprocess
from pathlib import Path
from setuptools import setup


def ensure_submodules():
    if Path(".gitmodules").exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
            )
        except Exception as e:
            # Don't hard-fail install if git is missing; just skip
            print(f"Warning: could not update submodules: {e}")


def parse_requirements(path: str):
    path = Path(path)
    requirements = []

    def _parse(file_path: Path):
        for line in file_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                included = line.split(maxsplit=1)[1]
                _parse(file_path.parent / included)
                continue

            # Normalize git-based requirements to valid PEP 508
            if "git+" in line:
                s = line
                # drop editable flag if present
                if s.startswith("-e "):
                    s = s[3:].strip()

                url = s
                name = None

                if "#egg=" in s:
                    url, egg = s.split("#egg=", 1)
                    url = url.strip()
                    name = egg.strip()
                else:
                    url = s.split()[0]
                    name = url.rsplit("/", 1)[-1]
                    if name.endswith(".git"):
                        name = name[:-4]

                requirements.append(f"{name} @ {url}")
            else:
                requirements.append(line)

    _parse(path)
    return requirements


ensure_submodules()
install_requires = parse_requirements("requirements.txt")

setup(
    name="world_engine",
    version="0.0.1",
    packages=[
        "world_engine",
        "depth_anything_v2",
        "owl_wms",
        "owl_vaes",
    ],
    package_dir={
        "world_engine": "src",
        "depth_anything_v2": "submodules/Depth-Anything-V2/depth_anything_v2",
        "owl_wms": "submodules/owl-wms/owl_wms",
        "owl_vaes": "submodules/owl-wms/owl-vaes/owl_vaes",
    },
    install_requires=install_requires,
)
