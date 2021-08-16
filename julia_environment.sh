# !/bin/sh

# This file activates and instantiates a julia virtual environment containing
# the required packages for workflow.sh.
# Must be run before workflow.sh

julia --project -e 'using Pkg; Pkg.activate()'

julia --project -e 'using Pkg; Pkg.instantiate()'
