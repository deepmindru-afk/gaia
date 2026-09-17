# from beartype.claw import beartype_this_package

# Nothing is imported here on purpose. The stackprinter excepthook moved to the
# entrypoints (app/main.py, app/worker.py): importing it here pulled numpy
# (86 modules, ~0.5 s) into every importer of every app module.

# beartype_this_package()
