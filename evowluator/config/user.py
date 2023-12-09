class User:
    """User config namespace."""
    BASE_PACKAGE = 'evowluator.user'
    PACKAGES = [f'{BASE_PACKAGE}.probes', f'{BASE_PACKAGE}.reasoners', f'{BASE_PACKAGE}.tasks']
