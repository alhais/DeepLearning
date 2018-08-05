# Get dataset and code 

git clone https://github.com/alhais/DeepLearning.git

cd DeepLearning

unzip -d datasets/video_jpegs/ datasets/video_jpegs.zip

## Generate edge images
python EdgeTransformation.py


zip -r myfiles.zip datasets/

# Get files from Colab Notebook
```ruby
from google.colab import files
import os
owd = os.getcwd()
os.chdir(os.getcwd() + '/datasets/video_jpegs/edge_videos/')
files.download('edge_video-2018-08-03-14-11-16 001.jpg')
os.chdir(owd)
```
