# Self_supervised_ACL

This git repo presents the pytorch-based implementation of the self-supervised angular contrastive loss from paper [*"Self-supervised learningof audio representations using angular contrastive loss"*](https://arxiv.org/abs/2211.05442). This repo is forked from the original author's repo [here](https://github.com/edufonseca/uclser20). The only difference is that we changed the training objectives. The motivation of applying ACL is demonstrated [here](https://github.com/shanwangshan/problems_of_infonce), the supervised ACL is implementated [here](https://github.com/shanwangshan/supervised_ACL), and the feature quality analysis is presented [here](https://github.com/shanwangshan/uniformity_tolerance). The procedure of preparing the data and creating features are strictly followed from the original authors.

## Training

To run the training script for different alpha values, here we submit an array job shown as below

`` sbatch gpu_sbatch.sh ``

The traning loss is angular contrastive loss (ACL) shown as below,

$ACL = \alpha * L1 + (1-\alpha) * L2$, where L1 is conventional contrastive loss (InfoNCE) and L2 is angular margin loss.

## Testing

The testing script is also the same as the original author.

`` python main_train.py -p config/params_supervised_lineval.yaml ``

However, we notice that the test script reports the macro-accuracy and the numbers are slightly different from what the authors reported in their paper. However, the micro-accuracy aligns with the numbers reported in the paper. Hence, in our paper, we report the micro-accuracy performance.

## Results

The results of applying ACL loss with respect to different $\alpha$ values are listed below,

| alpha | linear     |
|-------|------------|
| 0     | 71.295     |
| 0.1   | 74.994     |
| 0.2   | 74.701     |
| 0.3   | **77.070** |
| 0.4   | 71.712     |
| 0.5   | 74.013     |
| 0.6   | 76.596     |
| 0.7   | 75.671     |
| 0.8   | 76.472     |
| 0.9   | 74.453     |
| *1.0* | *74.160*   |

# Acknowledgement
This git repo is adapted from Fonseca et al. [here](https://github.com/edufonseca/uclser20).
