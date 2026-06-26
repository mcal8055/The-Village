"""Root conftest. Its mere presence puts the repo root on sys.path for pytest,
so test modules under scripts/ can `import paths` regardless of where pytest is
invoked from. Intentionally empty otherwise.
"""
