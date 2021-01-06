# Machine-Learning-Stats-Project
GMIT HDip Data Analytics 2020 Machine Learning and Statistics Project  
   
Use machine learning algorithms to provide predictions of wind turbine power output, based on a wind speed value
input by the user.  
    
A web page interface allows the user to enter values and returns predicted power output.  
The dataset that predictions are based on is held in a single csv file (source unknown), with 500 entries of  
wind speed and corresponding power output.  
This is read by a Python Flask server, and the data used to generate predictions by the Keras Neural Network
and Sklearn polynomial regression models.  
The source data contains zero power values below a certain minimum wind speed value, and above a certain maximum wind speed value.  
These are due to a minimum wind speed being required to overcome frictional forces in the mechanism, and turbines being shut down above 
a certain maximum wind speed to protect components from damage. 
There are also zero power values between the min and max mentioned, presumably due to turbines being offline for maintenance or if the local 
power network is already at full capacity.  
  
The web page that acts as the user interface displays these min and max values to the user, and an input box for them to enter a wind speed value.  
Four predicted power outputs are returned - two from each algorithm; one based on a dataset that excludes the zero power values associated with turbine down time, and one that includes these values.  

Files for building a Docker container are provided, for implementing the utility in a Cloud environment.  

The utility consists of the following (all in the same folder except 'static.getPower.html') :  
  
static/getPower.html  - web page interface  
.dockerignore         - list of files and directories for Docker to eclude from the build process  
.gitignore            - list of files not tracked by Git  
Dockerfile            - commands to build a Docker container and run the utility  
getPower.py           - Pythgon Flask server that loads the dataset, builds the machine learning models and responds to requests from the web page
LICENSE               - the Git license file
Machine Learning and Stats Project.ipynb  - jupyter notebook explaining and demonstrating how the utility works  
powerproduction.txt   - csv file containing source dataset: wind speed and associated power output  
Readme.md             - this file  
requirements.txt      - list of files required to build Docker container  
  
To run the utility locally : 
 
Linux :  
export FLASK_APP=getPower.py  
python3 -m flask run
  
Windows :  
set FLASK_APP=getPower.py  
python -m flask run  

To build/run the Docker container :  
docker build . -t getPower-image  
docker run --name getPower-container -d -p 5000:5000 getPower-image

Then use the url : http://127.0.0.1:5000/ in the browser navigation bar.  

Note :  
When the Python Flask server starts warnings may be output regarding GPUs if the machine it is being run on does not have GPUs - these can be ignored.
