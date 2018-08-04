# Get dataset and code 

git clone https://github.com/alhais/DeepLearning.git

cd DeepLearning

unzip -d datasets/video_jpegs/ datasets/video_jpegs.zip

## Generate edge images
python EdgeTransformation.py

# Get files from Colab Notebook
from google.colab import files

files.download('edge_video-2018-08-03-14-11-16 001.jpg')
