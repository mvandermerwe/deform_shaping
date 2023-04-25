import setuptools

setuptools.setup(
    name="deform_shaping",
    version="0.0.1",
    packages=["deform_shaping"],
    url="https://github.com/mvandermerwe/deform_shaping",
    description="Diff MPM for Deformable Object Shaping",
    install_requires=[
        "numpy", "torch"
    ]
)