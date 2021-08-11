Standardized process for generating reproducible code as described by Dr. Williams in our meeting on 8/4/21:

1. Put source code into src/Code_Snapshot
1a. Running files should conform to RunFile_<filename>.jl
1b. Put helper functions in different files (conforming to HelpFile_<filename>.jl) to minimize lines of code in runfiles

2. Put Data in nlpdrums/Data folder

3. Save outputs to Snapshot_Outs (create new subfolder if you'd like)


4. Put instructions for running code into workflow.sh so it can in theory all be executed at once

Julia boys: Make sure your julia environment is initalized at some point before running all scripts.





