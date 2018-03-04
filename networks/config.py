MODEL_CONFIG = {
    'actor': {
        'layers': [256, 256, 256],
        'learning_rate': 1e-4,
    },
    'critic': {
        'layers': [256, 256, 256],
        'learning_rate': 1e-3,
        'l2_rate': 1e-2,
    },
    'tau': 1e-3,
}
