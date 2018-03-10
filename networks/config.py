MODEL_CONFIG = {
    'actor': {
        'layers': [396, 396, 396],
        'learning_rate': 1e-4,
    },
    'critic': {
        'layers': [396, 396, 396],
        'learning_rate': 1e-4,
        'l2_rate': 1e-2,
    },
    'tau': 1e-3,
}
