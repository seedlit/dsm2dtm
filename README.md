<img align="right" width = 200 height=80 src="./data/logo.png">

# dsm2dtm

This repo generates DTM (Digital Terrain Model) from DSM (Digital Surface Model).

#### Example1: Input DSM and generated DTM over a flat terrain
![example](./results/result.png)

#### Example2: Input DSM, generated DTM, and groundtruth DTM (Lidar derived) over a hillside terrain
DSM was derived from [this point cloud data](https://cloud.rockrobotic.com/share/f42b5b69-c87c-4433-94f8-4bc0d8eaee90#lidar)
![example](./results/example2_dsm2dtm_hillside.png)

#### Installation (These step are for Linux. This will differ a bit for MacOS and windows)
Step 1: Clone the repo - `git clone https://github.com/seedlit/dsm2dtm.git` <br/>
Step 2: Move in the folder - `cd dsm2dtm` <br/>
Step 3: Create a virtual environment - `python3 -m venv venv` <br/>
Step 4: Activate the environment - `source venv/bin/activate` <br/>
Step 5: Install requirements - `pip install -r requirements.txt` <br/>
Step 6: Install saga_cmd - `sudo apt update; sudo apt install saga` <br/>

#### Usage
Run the script dsm2dtm.py and pass the dsm path as argument. <br/>
Example: `python dsm2dtm.py --dsm data/sample_dsm.tif` <br/> <br/>
Example with parameters: `python main.py --dsm sample_data/sample_dsm.tif --rgb sample_data/sample_rgb_ortho.tif --num_processes 2 --start_elev 19500 --end_elev 19700 --step_size 5 --gif_duration 20 --opaquenes 0.7`
