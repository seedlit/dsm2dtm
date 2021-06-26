<img align="right" width = 200 height=80 src="./data/logo.png">

# dsm2dtm

This repo generates DTM (Digital Terrain Model) from DSM (Digital Surface Model).

#### Example: Input DSM and generated DTM
![example](./results/result.png)

#### Installation (These step are for Linux. This will differ a bit for MacOS and windows)
Step 1: Clone the repo - `git clone https://github.com/seedlit/dsm2dtm.git` <br/>
Step 2: Move in the folder - `cd dsm2dtm` <br/>
Step 3: Create a virtual environment - `python3 -m venv venv` <br/>
Step 4: Activate the environment - `source venv/bin/activate` <br/>
Step 5: Install requirements - `pip install -r requirements.txt` <br/>
Step 6: Install saga_cmd - `sudo apt update; sudo apt install saga` <br/>

#### Usage
Run the script dsm2dtm.py and pass the dsm path as argument. <br/>
Example: `python dsm2dtm.py --dsm data/sample_dsm.tif`
