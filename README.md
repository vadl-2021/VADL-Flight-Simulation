# VADL-Flight-Simulation

## First Time Setup

1. Download Anaconda to use the Python code.
Once you have Anaconda, Spyder is the application that can be used to run the code.

2. Open Powershell Prompt and use the following commands to install the necessary packages:

conda install numpy

conda install pandas

conda install matplotlib

conda install openpyxl

## Description of Package Contents
### Python Scripts
- **MAIN.py**: Before calling this script, you must edit the configuration files (explained later) and set the relevant variables to the names of those files in the parameters.py script. The MAIN.py script is the code you will call to run the rocket flight simulation. Here, you can specify the name of the folder (_plot_folder_) to save your plots and flysheet in for a given simulation. There are also several variables that you can set to **True** or **False** to tell the code which plots you want to generate (e.g. _plot_x_z_), as well as booleans to specify whether you want to generate a flysheet and/or animation of the simulated rocket flight.
- **parameters.py**: This script will accept the names of your rocket, motor, and launch configuration (.yaml) files in the variables _rocket_config_, _motor_config_, and _launch_config_, respectively. It will convert the data from those files into Python variables, which can then be used to run the flight simulation.
- **calculate_trajectory.py**: This script is called by MAIN.py and wind_speed_analysis.py and performs the iterative calculations needed to complete the rocket flight simulation. It consists of 6 phases:
1. Powered Ascent on Launch Rail (**i_LRE** is the index to access values at the end of this phase)
2. Off-Rail Powered Ascent (**i_MECO** is the index to access values at the end of this phase)
3. Coast (Unpowered) Ascent (**i_apogee** is the index to access values at the end of this phase)
4. Free Fall Descent (**i_drogue** is the index to access values at the end of this phase)
5. Drogue Parachute Descent (**i_main** is the index to access values at the end of this phase)
6. Main Parachute Descent (**i_land** is the index to access values at the end of this phase)
- **wind_speed_analysis.py**: Before calling this script, you must edit the configuration files (explained later) and set the relevant variables to the names of those files in the parameters.py script. This script works similarly to MAIN.py, but it is specifically for simulating the same launch under a variety of wind conditions. You can specify an array of wind speeds (using the variable _wind_speeds_, in MPH) to simulate, after which flight data will be printed for each case and a plot will be generated showing the rocket's trajectory under each wind condition. 
- **draw_rocket.py**: This script generates a visualization of the rocket and motor's geometries and shows stability information (center of gravity and center of pressure). It can be called directly or indirectly from MAIN.py by setting _draw_rocket_ to **True**.
- **functions.py**: This script contains a variety of functions that are used elsewhere in the code for the flight simulation calculations.

### Folders
- **Flights**: This is where subfolders containing flight simulation data, plots, and flysheets are saved. The name of the subfolder for a given simulation is specified in MAIN.py.
- **Configuration Files**: This is where rocket, motor, and launch configuration files (.yaml files) are stored to be used by the code. It is important to keep each type of configuration file in its corresponding subfolder for the code to function properly. The example .yaml file for each type of configuration file can be copied and modified for a given set of simulation conditions.
1. **Rocket Configuration Files**: Consists of inputs to specify parameters for a given rocket. The topmost variable, _rocket_data_file_, corresponds to a file name for an Excel spreadsheet (found in the **Rocket Section Data** folder) consisting of rocket section lengths and masses (further explained later).
2. **Motor Configuration Files**: Consists of inputs to specify motor geometry and mass parameters, as well as thrust curve data which can generally be found at https://www.thrustcurve.org/. The topmost variable, _thrust_data_file_, consists of the name of a .csv file (found in the **Motor Data** folder) containing thrust curve data (further explained later).
3. **Launch Configuration Files**: Consists of variables for computational parameters (time step, maximum allowable simulation time), parachute deployment specifications, parachute failure overrides (to test a flight under drogue and/or main parachute failure conditions), launch rail parameters, and launch site conditions.
- **Rocket Section Data**: Contains Excel files specifying rocket section masses and lengths. The name of this file must be specified in the **rocket configuration** file. The first column contains the name of the sections (starting with the nose cone), and is only for reference. The second column contains the mass of each section (in lbs) and the third section contains the length of each section (in inches). Refer to the example spreadsheet to see this format.
- **Motor Data**: Contains .csv (comma separated variables) files of motor thrust data. The name of this .csv file must be specified in the **motor configuration** file. These can generally be obtained from https://www.thrustcurve.org/. The first column contains the time stamps of the motor thrust curve (in seconds) and the second column contains thrust values (in Newtons). The third column is optional, and contains the instantaneous motor propellent masses (in kilograms). Some thrust curve .csv files include this data -- if one does not, the mass is estimated linearly from the initial motor mass specified in the **motor configuration** file. Refer to the example .csv file to see this format.
