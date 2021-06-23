GAIA-cv
^^^^^^


Introduction 
------------
GAIA-cv is a fundamental library for customized computer vision application and supports many projects as follows:

- GAIA-cls: Coming soon!
- GAIA-det_: GAIA detection toolbox that provides automatic AI solutions for object detection.
- GAIA-seg: Coming soon!
- GAIA-ssl: Coming soon!

.. _GAIA-det: https://github.com/GAIA-vision/GAIA-det

It provides functionalities that help the customization of AI solutions.

- Design and apply customized search space of any type with little efforts.
- Manage models in search space according to your rules.
- Integrate datasets of various sources.


Installation
------------

- Install the full version of mmcv, please refer to here_.
- Install gaiavision in two lines:
  .. code-block:: shell
  git clone https://github.com/GAIA-vision/GAIA-cv . && cd GAIA-cv
  pip install -e .


.. _here: https://github.com/open-mmlab/mmcv#installation


Acknowledgements
---------------

- This repo is constructed on top of the awesome mmcv_.
- Some of the core ideas are inspired by an amazing work BigNas_.




.. _mmcv: https://github.com/open-mmlab/mmcv
.. _BigNas: https://arxiv.org/abs/2003.11142


Contributors:
-------------

We encourage every practitioners in the community to contribute to GAIA-vision. The coutributions including but not limited to implementing new search spaces, new architectures, customized solutions on public datasets and etc. are welcomed. Please refer to CONTRIBUTION.md. For now, the contributors include Junran Peng, Xingyuan Bu, Qing chang, Haoran Meng.
