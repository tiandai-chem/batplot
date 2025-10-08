"""batplot package."""

from importlib.metadata import version, PackageNotFoundError

try:
	__version__ = version("batplot")
except PackageNotFoundError:
	# During local dev / editable installs before metadata exists
	__version__ = "1.0.14"

__all__ = ["__version__"]
