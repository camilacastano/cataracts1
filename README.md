# First implementation of segmentation

The goal of this first development is to use different colour masks and detect cataract fragments during the surgery phase.

## Cataract fragment detection (ID tracking)

This is the first task of my thesis: to be able to track cataract fragments and follow them throughout the surgery. We will have bounding boxes for each fragment, persistent IDs (follow them all surgery phase)

For this script is important to do the correct dependencies installation, debugging of a single frame, full video processing, and analysis of the final CSV (useful for our protocol)

### June 4th
For now, I've been implementing libraries that already contain pre-trained segmentation models, but due to version differences, I couldn't continue searching for another tool. 

I'm looking for certain models and reading their documentation on how the training was done, before implementing my own ideas. Important to do a course for this specifically


## Updates

### June 9th
Today will be implementing a model called VideoMamba for video segmentation on more time-consuming surgeries. Which follows ViT (vision transformers) standard architecture

VideoMamba refers to a class of neural architectures using structured state-space models for video sequence modeling and understanding. The idea is to clone the repository with all its dependencies and test our pre-dataset with it to see if it functions.
