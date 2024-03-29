GAIA-cv
^^^^^^


Introduction 
------------
GAIA-cv is a fundamental vision library for customized AI solutions, and supports many projects as follows:

- GAIA-det_: GAIA toolbox that provides automatic AI solutions for object detection.
- GAIA-ssl: GAIA toolbox that provides automatic AI solutions for self-supervised learning.
- GAIA-seg: GAIA toolbox that provides automatic AI solutions for semantic segmentation. Almost ready, coming soon!

.. _GAIA-det: https://github.com/GAIA-vision/GAIA-det

It provides functionalities that help the customization of AI solutions.

- Design and apply customized search space of any type with little efforts.
- Manage models in search space according to your rules.
- Integrate datasets of various sources.

Requirements
------------
- Python 3.6+
- CUDA 10.0+
- mmcv >= 1.2.7 & mmcv < 1.3.0 
- Others (See requirements.txt)

Installation
------------

- Install the full version of ``mmcv``. Please refer to this_.
- Install ``gaiavision`` in three lines:

.. code-block:: bash

  git clone https://github.com/GAIA-vision/GAIA-cv . && cd GAIA-cv
  pip install -r requirements.txt
  pip install -e .


.. _this: https://github.com/open-mmlab/mmcv#installation

Usage
-----
.. code-block:: python

  import gaiavision # load registered modules
  


Acknowledgements
---------------

- This repo is constructed on top of the awesome toolbox mmcv_.
- Some of the ideas are inspired by an amazing work named BigNAS_.




.. _mmcv: https://github.com/open-mmlab/mmcv
.. _BigNas: https://arxiv.org/abs/2003.11142


Contributors
-------------

We encourage every practitioner in the community to contribute to GAIA-vision. The coutributions including but not limited to implementing new search spaces, new architectures, customized solutions on public datasets and etc. are welcomed. For now, the contributors include Junran Peng, Xingyuan Bu, Qing Chang, Haoran Yin.

