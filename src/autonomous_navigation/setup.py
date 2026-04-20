import os
from glob import glob

from setuptools import setup

package_name = "autonomous_navigation"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    package_dir={package_name: "."},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fedir",
    maintainer_email="fedir@todo.todo",
    description="Autonomous navigation ROS 2 package.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "autonomous_navigation = autonomous_navigation.main_node:main",
        ],
    },
)
