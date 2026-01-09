This multi-task CNN is designed to analyze food-related images from social media advertisements using multiple visual classification tasks.<br>

A multi-task approach was chosen to evaluate how the model performs when learning multiple visual tasks simultaneously using a shared backbone.<br>
<br>

**Head-1**: Multi-class classification 
- Social media channel prediction
- Accuracy: 96.13%

**Head-2**: Multi-label classification 
- Creator type prediction
- Micro F1 score: 0.92

**Head-3**: binary classification 
- Logo presence (yes/no)
- Positive-Class F1-score: 0.94<br>
<br>

Dataset: 5851 images -> training 70%, validation 15%, testing 15%

Model overview (see below):

![alt](https://github.com/IoannisKotsis/CNN/blob/1d9daf32bab2256a92a802298c9db6224521edaa/cnn_structure.png)
