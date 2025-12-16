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


ensure_submodules()
install_requires = parse_requirements("requirements.txt")

setup(
    name="world_engine",
    version="0.0.1",
    packages=[
        "world_engine",
        # "depth_anything_v2",
        # "depth_anything_v2.dinov2_layers",
        # "depth_anything_v2.util",
        "owl_vaes",
        "owl_vaes.models",
        "owl_vaes.utils",
        "owl_vaes.nn",
        "owl_wms",
        "owl_wms.models",
        "owl_wms.nn",
    ],
    package_dir={
        "world_engine": "src",

        # "depth_anything_v2": "submodules/Depth-Anything-V2/depth_anything_v2",
        # "depth_anything_v2.dinov2_layers": "submodules/Depth-Anything-V2/depth_anything_v2/dinov2_layers",

        "owl_vaes": "submodules/owl-wms/owl-vaes/owl_vaes",

        "owl_wms": "submodules/owl-wms/owl_wms",
        "owl_wms.models": "submodules/owl-wms/owl_wms/models",
        "owl_wms.nn": "submodules/owl-wms/owl_wms/nn",
    },
    install_requires=install_requires,
)
