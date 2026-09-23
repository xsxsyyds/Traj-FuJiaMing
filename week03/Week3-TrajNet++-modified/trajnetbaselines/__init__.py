__version__ = '0.1.0'

from . import augmentation
from . import lstm
from . import sgan
# Classroom patch: classical baselines are optional; import explicitly if needed.
from . import vae
