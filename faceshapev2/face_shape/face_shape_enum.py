from enum import Enum

class FaceShape(Enum):
    OVAL = "Oval"
    ROUND = "Round"
    SQUARE = "Square"
    HEART = "Heart"
    DIAMOND = "Diamond"
    OBLONG = "Oblong"
    
    @staticmethod
    def get_description(shape):
        descriptions = {
            FaceShape.OVAL: "Oval face shape is considered the most balanced.",
            FaceShape.ROUND: "Round face shape has similar length and width with soft angles.",
            FaceShape.SQUARE: "Square face shape has strong jaw and angular features.",
            FaceShape.HEART: "Heart face shape is wider at forehead and narrow at chin.",
            FaceShape.DIAMOND: "Diamond face shape has high cheekbones and narrow forehead/jawline.",
            FaceShape.OBLONG: "Oblong face shape is longer than it is wide."
        }
        return descriptions.get(shape, "Unknown face shape")