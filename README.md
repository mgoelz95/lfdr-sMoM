# lfdr-sMoM

Spatial Multiple Hypothesis Testing with Local False Discovery Rates (spatialmht)
==========================================

The `spatialmht` package provides an implementation of the multiple hypothesis testing methods for spatial data introduced in the references below. This empirical Bayes method estimates the local false discovery rates (lfdr's) with the spectral method of moments (sMoM) and allows to assert the state of a spatial field at the sensors and in between sensors. The method identifies anomalies within the observation area with false positive control (w.r.t. the False Discovery Rate).

Improvements of the base-line method lfdr-sMoM from [Goelz2022TISPN] were proposed in subsequent works [Goelz2022CISS] and [Goelz2022ICASSP]. These are also implemented in this package. The details on the proposed methods are found in the references given below.

Installation
------------

To install the package:

```
pip install spatialmht
```

Then go to with_pip_install and run file main.py in this directory. If you do not want to use pip to install the package, you can simply download this repository and run main.py from directory without_pip_install.

Example
------------------

The provided file `main.py` allows you to recreate the essential plots from the three references given below. By modifying `parameters.py`, you also have the opportunity to create new spatial fields that similate the propagation of radio waves according to your own desires (with custom number of transmitters, sensors, observations per sensor, propagation environment, sensor densities etc). For such fields, observations and p-values are generated using an energy detector at each sensor. In addition, you can also provide your own p-values generated using your own methods with sensors located at spatial locations that you provide! 

If your data does not have any particular spatial structure, or maybe is not even spatial data at all, you can still apply lfdr-sMoM with your custom p-values. However, the spatial interpolation of the lfdrs and spatially varying priors will not be performed in such a case. Please see the comments in `main.py` to learn about the possible ways you can modify the file to suit your own needs.



References
----------

[Goelz2022TISPN]: **Multiple Hypothesis Testing Framework for Spatial Signals**. M. Gölz, A.M. Zoubir and V. Koivunen, IEEE Transactions on Signal and Information Processing over networks, July 2022, [DOI:10.1109/TSIPN.2022.3190735](https://ieeexplore.ieee.org/abstract/document/9830080).

[Goelz2022CISS]: **Estimating Test Statistic Distributions for Multiple Hypothesis Testing in Sensor Networks** M. Gölz, A.M. Zoubir and V. Koivunen, 2022 56th Annual Conference on Information Sciences and Systems (CISS), Princeton, NJ, February 2022, [10.1109/CISS53076.2022.9751186](https://ieeexplore.ieee.org/abstract/document/9751186).

[Goelz2022ICASSP]: **Improving Inference for Spatial Signals by Contextual False Discovery Rates**. M. Gölz, A.M. Zoubir and V. Koivunen, 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP2022), Singapore, [DOI:10.1109/ICASSP43922.2022.9747596](https://ieeexplore.ieee.org/abstract/document/9747596).