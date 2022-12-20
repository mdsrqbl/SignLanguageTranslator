# Sign Language Translator

## Overview
Sign language consists of gestures and expressions used mainly by the hearing-impaired to talk. This project is an effort to bridge the communication gap between the hearing and the hearing-impaired community using Artificial Intelligence.

The goal is to provide a user friendly api like HuggingFace to novel Sign Language Translation solutions.

#### Major Components and Goals ####
1. Sign language to Text
    - In speech to text translation, features such as mel-spectrograms are extracted from audio and fed into neural networks which then output text tokens corresponding to what was said in the audio.
    - Similarly, pose vectors (2D or 3D) are extracted from video, and to be mapped to text corresponding to the performed signs, they are fed into a neural network which is a checkpoint of a SOTA speech-to-text model finetuned using gradual unfreezing starting from the layers near input towards the output layers.

2. Text to Sign Language
    - This is a relatively easier task as it can even be solved with HashTables. Just parse the input text and play approproate video clip for each word.

    1. Motion Transfer
        - This allows for seamless transitions between the clips. The idea is to concatenate pose vectors in the time dimention and transfer the movements onto any given image of any person.
    2. Pose Synthesis
        - This is similar to speech synthesis. It solves the challenge of unknown synonyms or hard to tokenize/process words/phrases.
        - It can also be finetuned to make avatars move in desired ways using only text.

3. Preprocessing Utilities
    1. Pose Extraction
        - Mediapipe 3D world coordinates and 2D image coordinates
        - Pose Visualization
    2. Text normalization
        - Since the supported vocabulary is handcrafted, unknown words (or spellings) must be substituted with the supported words.

## How to install the package
    - git clone https://github.com/mdsrqbl/SignLanguageTranslator.git
    - cd SignLanguageTranslator
    - pip install -e .

- pip install git+https://github.com/mdsrqbl/SignLanguageTranslator.git

## Package Architecture
    SignLanguageTranslator
    ├── SignLanguageTranslator
    │   ├── datasets
    │   │   ├── signs_recordings
    │   │   │   ├── Landmarks
    │   │   │   └── Videos
    │   │   │
    │   │   └── texts_parallel_corpus
    │   │
    │   ├── text
    │   ├── utils
    │   │   └── dataCollection
    │   │       ├── text
    │   │       └── video
    │   │
    │   └── vision
    │
    ├── notebooks
    │   ├── inputs
    │   └── outputs
    │
    └── tests

## Datasets
- Pakistani Sign Language Clips: 840 clips performed by 12 people (10 normies, 2 hearing impaired).
    - Labeled in Urdu & English.
- Text2Text: Spoken Language Text to Sign Language Text (restructure & shorten) (both Urdu & English).
- English & Urdu Text Corpora used in synthethetic data generation (contain all and only the supported words).

## Roadmap
- [x] Old project setup on Git
- [-] Landmarks Utils
    - [x] LandmarksInfo
        - [x] counts
        - [x] connections
        - [x] colors
        - [x] reshaper
    - [x] Visualization (3D)
        - [x] single frame image
        - [x] multiple frames on one image
        - [x] graph video
    - [x] 4D Transformations
        - [x] zoom
        - [x] rotation
        - [x] noise
        - [x] time dialation
        - [x] stabilization/rectification
            - [x] warp video
    - [x] concatenate clips (video/landmarks)
        - [x] transitions
        - [x] trimming
    - [ ] test cases
- [ ] text preprocessing
    - [ ] normalize text
    - [ ] tokenize / phrases (spaCy matcher for )
    - [ ] embeddings
    - [ ] word error rate / similarity metrics etc
- [ ] Video data
    - [ ] Cut raw videos
    - [ ] completeness check
    - [ ] filenames
    - [ ] pose extraction
    - [ ] upload / sync
- [ ] text Utils
    - [ ] list supported words
    - [ ] mappers (word2word, word2embedding)
    - [ ] text2text dataset

    - [ ] text corpora mining/scrapping

- [ ] Development
    - [ ] Website integration with package
    - [ ] send synthesized video directly from memory (no IO/HttpStreaminResponse)
    - [ ] recorded video fps/preprocessing progress bar
    - [ ] seperate thread for video preprocessing
    - [ ] append synthesized to drop down instead of overwriting
    - [ ] + button --> dropdown of languages --> append text editor (popped from dropdown) --> has x button except for last, also change language button

- [ ] Data loader
    - [ ] data loader class
    - [ ] text to pose/clip
    - [ ] batch generator (pytorch)
- [ ] Training ( || research paper)
    - [ ] whisper --> gesture :)
    - [ ] finetune mediapipe
    - [ ] T5 (summarizer) on text2text dataset
    - [ ] pose generation (text 2 audio)
    - [ ] motion transfer (.../stable diffusion)
    - [ ] pose to image (GAN/stable diffusion)
- [ ] Deployment
    - [ ] API / MLOps
    - [ ] finalize dataset structure/storage
- [ ] Web Development
    - [ ] Website integration with models/package
    - [ ] data collection page/app
        - [ ] reference clips loading
        - [ ] timestamping for continuous recording
        - [ ] video annotation
    - [ ] front end
- [ ] Show off results to Hamza Foundation
    - [ ] text restructuring dataset
    - [ ] sign language videos dataset
- [ ] Training ( || research paper)
    - [ ] activity recognition/generation
    - [ ] better embedding model
- [ ] ★ Publish Research Paper & Datasets ★
- [ ] Raise Awareness, Hand over the project.
- [ ] Start living life again.

## Todo:
- add a clip info table, fps resolution etc
- test reshaper on single frame
- test scale_landmarks in T.stabilize_clips()
### monitering
- average input length
- average image brightness
- average hidden landmarks
- average hand position/distances
- number of times null output
- number of time quick consequtive similar input
- when user quits
- CTR
- Dashboard
- activity detection --> trim video
## Research Paper

## Credits and Gratitude


## Bonus