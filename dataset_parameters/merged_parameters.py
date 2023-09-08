"""Original parameters."""

n_classes = 5
# cls_weights = [0.0, 0.5, 1.31237, 1.38874, 1.39761, 1.5, 1.47807]
cls_weights = []
cls_names = ['unlabeled', 'Forest', 'Plain', 'Building', 'Road']
label_colors = {
    0: (0, 0, 0),  # unlabeled
    1: (16, 128, 64),  # Forest
    2: (0, 128, 128),  # Plain
    3: (0, 0, 192),  # Building
    4: (255, 255, 255)  # Road
}
