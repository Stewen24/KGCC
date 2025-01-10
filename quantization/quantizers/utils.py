class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not been initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__(
            "Quantizer has  not been initialized yet"
        )
