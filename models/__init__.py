from models.model_PR_PL_quantized import pr_pl_quantized
from utils import ClassEnumOptions, MethodMap


class QuantArchitectures(ClassEnumOptions):
    pr_pl_quantized = MethodMap(pr_pl_quantized)
