## Generate Train Data
```ruby
!pip install git+https://www.github.com/keras-team/keras-contrib.git
!pip install python_speech_features
%cd /content
!rm -r DeepLearning
!git clone https://github.com/alhais/DeepLearning.git
!python DeepLearning/tools/create_dataset.py
```
## Train GAN
```ruby
%cd /content/DeepLearning/pix2pix/
!python pix2pix.py
```
## Check Results
```ruby
%cd /content/DeepLearning/pix2pix/images/facades
!ls
from google.colab import files
files.download('1_0.png')
```

# Get files from Colab Notebook
```ruby
from google.colab import files
import os
owd = os.getcwd()
os.chdir(os.getcwd() + '/DeepLearning/pix2pix/datasets/facades/train/B')
files.download('10_emg.jpg')
os.chdir(owd)
```
