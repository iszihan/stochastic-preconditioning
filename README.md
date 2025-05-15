# Stochastic Preconditioning for Neural Field Optimization [Siggraph 2025]
<img src="./assets/teaser_website_wistia.png" width="100%">

### [Project Page](https://research.nvidia.com/labs/toronto-ai/stochastic-preconditioning/) | [Paper](https://arxiv.org/abs/2006.09661) 

[Selena Ling](https://iszihan.github.io/),
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Alec Jacobson](https://www.cs.toronto.edu/~jacobson/),
[Nicholas Sharp](https://nmwsharp.com/)<br>
ACM Transaction on Graphics (Proceedings of SIGGRAPH North America 2025)

## Get started
You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```
## SDF from Oriented Point Cloud Experiment
We provide code for the SDF fitting from Oriented Point Cloud experiment (Section 5.1.1).

Run the following script for regular training without stochastic preconditioning:
```
hi
```

Add `--sp y` to train with stochastic preconditioning as following:
```
hi
```

## Credits 
This repo is built from existing codes from [Siren](https://github.com/vsitzmann/siren), [SDFStudio](https://github.com/autonomousvision/sdfstudio) and [PET-NeuS](https://github.com/yiqun-wang/PET-NeuS). We thank the maintainers for their contribution to the community!

<!-- ## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{sitzmann2019siren,
    author = {Sitzmann, Vincent
              and Martel, Julien N.P.
              and Bergman, Alexander W.
              and Lindell, David B.
              and Wetzstein, Gordon},
    title = {Implicit Neural Representations
              with Periodic Activation Functions},
    booktitle = {arXiv},
    year={2020}
}
``` -->

## Contact
If you have any questions, please feel free to email the authors.
