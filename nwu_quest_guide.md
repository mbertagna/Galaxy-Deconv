# Northwestern Quest Supercomputer Quickstart Guide

## Getting Started

### Logging In
To access Quest, use SSH with X forwarding:
```bash
ssh -X <netid>@quest.northwestern.edu
```
[Learn more about Quest Login](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1541)

### Finding Your Project Directory
If your class or lab has a project allocation, it is best to work on your projects there, as it likely has much more storage than your home directory which only has 80GB. If you do not know your project id you can find it with this command which finds all project directories you have access to:
```bash
for dir in /projects/*/; do [ -r "$dir" ] && [ -w "$dir" ] && echo "$dir"; done
```

Navigate to your specific project directory:
```bash
cd /projects/<project_id>/
```
[Learn more about Project Directories](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1542)

Set up your directories:
```bash
mkdir -p michael/git
cd michael/git
```

Clone repositories:
```bash
git clone https://github.com/mbertagna/Galaxy-Deconv.git
```

## Data Management

### Transferring Data
For large data transfers, compress files using pigz (parallel gzip) for better performance on your local machine:
```bash
tar cf - simulated_datasets/ | pv -s $(du -sb simulated_datasets/ | awk '{print $1}') | pigz -9 -p 4 > simulated_datasets.tar.gz
```

### Using Globus
For transferring data between institutions or large datasets:
[Learn more about Globus](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1962)

## Running Jobs with SLURM

### Creating a Job Script
We will run through an example where we decompress the compressed data file.

Create a file named `decompress_data.sh` with the following contents:
```bash
#!/bin/bash
#SBATCH --account=e32704  ## Required: your allocation/account name
#SBATCH --partition=short  ## Required: partition type
#SBATCH --time=04:00:00    ## Required: job runtime
#SBATCH --nodes=1          ## Number of nodes
#SBATCH --ntasks-per-node=8 ## Number of tasks per node
#SBATCH --mem=16G           ## Memory allocation per node
#SBATCH --job-name=decompress_data ## Job name

# Load pigz module
module load pigz/2.7-gcc-12.3.0  # Load the latest pigz module

# Extract the tar.gz archive
tar --use-compress-program=pigz -xvf /projects/e32704/michael/git/Galaxy-Deconv/simulated_datasets.tar.gz -C /projects/e32704/michael/git/Galaxy-Deconv/
```

### Managing Jobs
Submit a job:
```bash
sbatch decompress_data.sh
```

Check job status in the short partition:
```bash
squeue | grep short
```

Cancel a job:
```bash
scancel <jobid>
# Example: scancel 7958659
```
[Learn more about SLURM on Quest](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1964)

### GPU Jobs
Check GPU job status:
```bash
squeue | grep gengpu
```

## Software Environment

### Managing Python Environments
View available Python versions:
```bash
module avail python
```

Load a specific Python version:
```bash
module load python/3.10.1
python --version
```

### Useful File Editing Commands
Change paths in multiple files:
```bash
sed -i 's|/home/michaelbertagna/git/Galaxy-Deconv/|/projects/e32704/michael/git/Galaxy-Deconv/|g' train.py
```

### Text Editors in SSH

#### Nano (Beginner-friendly)
```bash
nano filename
```
- Easy to use with on-screen shortcuts

#### Vim (Advanced)
```bash
vim filename
```
- More powerful but steeper learning curve

## Helpful Aliases

Add these to your `.bashrc` file:
```bash
# Check recently modified files
alias moddate='ls -lt'

# View only your SLURM jobs with detailed formatting
alias myqueue='squeue --format="%.18i %.9P %.80j %.8u %.8T %.10M %.9l %.6D %R" --me'
```
