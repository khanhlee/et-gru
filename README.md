# ET-GRU: multi-layer gated recurrent units and position specific scoring matrices to identify electron transport proteins
<img src="fig/etc.png?sanitize=true" width=600 class="center">
Electron transport chain is a series of protein complexes embedded in the process of cellular respiration, which is an important process to transfer electrons and other macromolecules throughout the cell. It is also the major process to extract energy via redox reactions in the case of oxidation of sugars. Many studies have determined that the electron transport protein has been implicated in a variety of human diseases, i.e. diabetes, Parkinson, Alzheimer’s disease and so on. Few bioinformatics studies have been conducted to identify the electron transport proteins with high accuracy, however, their performance results require a lot of improvements. Here, we present a novel deep neural network architecture to address this problem.
Most of the previous studies could not use the original position specific scoring matrix (PSSM) profiles to feed into neural networks, leading to a lack of information and the neural networks consequently could not achieve the best results. In this paper, we present a novel approach by using deep gated recurrent units (GRU) on full PSSMs to resolve this problem. Our approach can precisely predict the electron transporters with the cross-validation and independent test accuracy of 93.5 and 92.3%, respectively. Our approach demonstrates superior performance to all of the state-of-the-art predictors on electron transport proteins.
Contain source code and dataset for re-implementing ET-GRU

## Implementation

File structure:
- electron.py - Source code for training model
- dataset - contains pssm profiles, list of cross-validation and independent dataset files
- model - ET-GRU final model

## Citation
Please cite our paper as:
>Le, N.Q.K., Yapp, E.K.Y. & Yeh, H.Y. ET-GRU: using multi-layer gated recurrent units to identify electron transport proteins. *BMC Bioinformatics* 20, 377 (2019). https://doi.org/10.1186/s12859-019-2972-5
