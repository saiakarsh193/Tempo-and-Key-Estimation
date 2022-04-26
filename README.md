# Tempo-and-Key-Estimation using CNN
## MMT Final Project
- #### [Sai Akarsh](https://github.com/saiakarsh193) (2019111017)
- #### [Nikit Uppala](https://github.com/Nikit-Uppala) (2019101022)

## Goal
We are implementing axis-aligned directional CNNs for estimating tempo and key. This is done by calculating the CQT spectrogram of the audio for key estimation and running it through the CNN and similarly calculating the Mel spectrogram of the audio for tempo estimation and running it through the CNN for training.

## Datasets chosen
- [Extended Ballroom](http://anasynth.ircam.fr/home/media/ExtendedBallroom) for Tempo Estimation Training
- [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) for Tempo Estimation Testing
- [Gainsteps](https://github.com/GiantSteps/giantsteps-key-dataset) for Key Estimation

## References
- Link to [paper](https://arxiv.org/pdf/1903.10839.pdf)
- Paper implementation GitHub [repository](https://github.com/hendriks73/directional_cnns)