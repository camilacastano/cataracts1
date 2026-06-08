# First implementation of segmentation

The goal of this first development is to use different colour masks and detect cataract fragments during surgery phase.

## Cataract fragment detection (ID tracking)

This is the first task of my thesis, to be able to track cataract fragments and follow them throughout the surgery. We will have bounding boxes for each fragment, persisten IDs (follow them all surgery phase)

For this script is important to do the correct dependencies installation, debugging of a single frame, full video processing and analysis of final CSV (useful for our protocol)

### June 4th
For now, I've been implementing certain librariesw that already contain pre-trained models related to segmentation, but with the difference between version, it was not possible to continue. Searching for another tool. 

I'm looking for certain models and read their documentation on ho the training was done, before implementing my own ideas. Important to do a course for tis specifically


## Updates

### June 9th
Today will be implementing a model called VideoMamba for video segmentation on more time consuming surgeries. Which follows ViT (vision transformers) standard architecture

VideoMamba refers to a class of neural architectures using structured state-space models for video sequence modeling and understanding