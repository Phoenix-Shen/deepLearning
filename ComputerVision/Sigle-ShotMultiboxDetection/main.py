
from model import SigleShotMultiboxDetection

if __name__ == "__main__":
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0])+len(ratios[0])-1
    model = SigleShotMultiboxDetection(
        "./dataset/banana-detection/",
        40,
        32,
        True,
        0.1,
        5e-4,
        1,
        num_anchors,
        sizes,
        ratios,
    )
    model.train()
