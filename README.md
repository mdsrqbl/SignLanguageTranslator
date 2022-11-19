# Sign Language Translator

## Overview
Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly api like HuggingFace to novel Sign Language Translation solutions.

#### Major Components and Goals ####
1. Sign language to Text
    - In speech to text translation, features such as mel-spectrograms are extracted from audio and fed into neural networks which then output text token corresponding to what was said in the audio. 
    - Similarly, pose vectors (2D or 3D) are extracted from video and to be be mapped to text corresponding to the performed signs, they are fed into a neural network which is a finetuned checkpoint of a SOTA speech to text model trained using gradual unfreezing starting from the layers near input towards the output layers.

2. Text to Sign Language
    - This is a relatively easier task as it can even be solved with HashTables. Just parse the input text and play approproate video clip for each word.
    
    1. Motion Transfer
        - This allows for seamless transitions between the clips. The idea is to concatenate pose vectors in the time dimention and transfer the movements onto any given image of any person.
    2. Pose Synthesis
        - This is similar to speech synthesis. It solves the challenge of unknown words.
        - It can also be finetuned to make avatars move in desired ways using only text.
        
3. Preprocessing Utilities
    1. Pose Extraction
        - Mediapipe 3D world coordinates and 2D image coordinates
    2. Text normalization
        - Since the supported vocabulary is handcrafted, unknown words (or spellings) must be substituted with the supported words.

## How to install the package
- git clone https://github.com/mdsrqbl/SignLanguageTranslator.git
- pip install git+https://github.com/mdsrqbl/SignLanguageTranslator.git

## Package Architecture
- 

## Research Paper

## Credits and Gratitude

