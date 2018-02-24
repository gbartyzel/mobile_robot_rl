MODEL_CONFIG = {
    'actor': {
        'layers': [400, 300, 300],
        'lrate': 1e-4,
    },
    'critic': {
        'layers': [400, 300, 300],
        'learning_rate': 1e-3,
        'l2_rate': 1e-2,
    },
    'tau': 1e-3,
}
