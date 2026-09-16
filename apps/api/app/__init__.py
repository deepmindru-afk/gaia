# from beartype.claw import beartype_this_package

# Nothing is imported here on purpose. The stackprinter excepthook this package
# used to install lives in the process entrypoints (app/main.py, app/worker.py)
# instead: importing stackprinter pulls numpy (86 modules, ~0.5 s), and putting
# it in the package __init__ charged that to every importer of every app module
# — including app.constants leaves, every test process and every mutant.

# beartype_this_package()
