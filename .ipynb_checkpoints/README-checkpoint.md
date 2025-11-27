# TrumpTwitterAnalysisMDS
Group 14 MDS 2025 DSCI 522 project.


## Environment and data analysis setup
To set up the necessary packages for this project, run the following from the root of the repository:
```bash
conda-lock install --name 522_proj conda-lock.yml
``` 
To open the data analysis, open the <file_name> in jupyter lab by running the following:
```bash
jupyter lab
```
And then open the  <file_name> .
To select the environment we just created, go to the top right corner and under "Select Kernel" choose "Python [conda env:522_proj]".
To run the data analysis, under "Run" select "Restart Kernel and run all cells".

## Dependencies
 - `conda` (version 25.9.1 or higher)
 - `conda-lock` (version 3.0.4 or higher)
 - `jupyter-lab` (version  4.4.7   or higher)
 - `nb_conda_kernels` (version 2.5.1 or higher)
 - All packages listed in [`environment.yml`](environment.yml)