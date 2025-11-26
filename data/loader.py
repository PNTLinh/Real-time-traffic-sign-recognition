from roboflow import Roboflow
from dotenv import load_dotenv
import os
load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("hust-iwlti").project("vietnam-traffic-sign-vr1a7-ecrhf-xasim")
version = project.version(3)
dataset = version.download("yolov12")

