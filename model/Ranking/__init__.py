# Non Personalized
from .TopPopular import TopPopular

# Non Neural model
## Matrix Factorization
from .WMF import WMF
from .BPRMF import BPRMF

from .EASE import EASE
from .SLIMElastic import SLIMElasticNet as SLIM
from .P3a import P3a
from .RP3b import RP3b
from .RecWalk import RecWalk

# Neural model
## Autoencoder
from .CDAE import CDAE
from .DAE import DAE
from .MultDAE import MultDAE
from .MultVAE import MultVAE
from .RecVAE import RecVAE

from .Dropout_DAE import Dropout_DAE
from .NDAE import NDAE
from .ND_MultDAE import ND_MultDAE
from .ND_MultVAE import ND_MultVAE
from .Denoising_EASE import Denoising_EASE
from .WMF_EASE import WMF_EASE
## 
from .NGCF import NGCF