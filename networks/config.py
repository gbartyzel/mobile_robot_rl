MODEL_CONFIG = {
    'actor': {
        'layers': [512, 512, 64],
        'learning_rate': 1e-4,
    },
    'critic': {
        'layers': [512, 512, 64],
        'learning_rate': 5e-4,
        'l2_rate': 1e-2,
    },
    'tau': 1e-3,
}
