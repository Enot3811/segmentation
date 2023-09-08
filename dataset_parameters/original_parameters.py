"""Original parameters."""

n_classes = 7
# cls_weights = [0.0, 0.5, 1.31237, 1.38874, 1.39761, 1.5, 1.47807]
cls_weights = [0.0, 0.3512100881284803, 2.0025812820597606, 2.490657048264174,
               2.8399131854794275, 6.116008441653059, 4.035081528002041]
cls_names = ['unlabeled', 'Dense forest', 'Sparse forest', 'Moor',
             'Herbaceous formation', 'Building', 'Road']
label_colors = {
    0: (0, 0, 0),  # unlabeled
    1: (16, 128, 64),  # Dense forest
    2: (192, 255, 64),  # Sparse forest
    3: (192, 192, 92),  # Moor
    4: (0, 128, 128),  # Herbaceous formation
    5: (0, 0, 192),  # Building
    6: (255, 255, 255)  # Road
}
