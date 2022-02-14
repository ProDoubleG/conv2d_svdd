class GlobalConfig:
    def __init__(self):
        self.BUNDLE_SIZE = 200
        self.FINAL_SPACE_DIMENSION = 24
        self.VALIDATION_RATIO = 0.2
        self.MODEL_TYPE = 'conv' # or dense
        self.VIOLATION_CONSTANT = 2
        
        self.FEATURE_NUMBER = 2