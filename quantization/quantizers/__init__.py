from quantization.quantizers.base_quantizers import QuantizerBase
from quantization.quantizers.uniform_quantizers import (
    SymmetricUniformQuantizer,
    AsymmetricUniformQuantizer,
)
from utils import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
