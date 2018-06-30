# GraphEmbedding

## Datasets

Get the datasets from https://drive.google.com/open?id=1lY3pqpnUAK0H9Tgjyh7tlMVYy0gYPthC
and extract under `data/`:
* AIDS10knef
* AIDS700nef
* AIDS80nef
* Web
* ?


## Dependencies

Install the following the tools and packages:

* `python3`: Assume `python3` by default (use `pip3` to install packages).
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `tensorflow`
* `networkx==1.10` (NOT `2.1`)
* `beautifulsoup4`
* `lxml`
* `matplotlib`
* `pytz`
* `pygraphviz`. The following is an example set of installation commands (tested on Ubuntu 16.04) 
    ```
    sudo apt-get install graphviz libgraphviz-dev pkg-config
    pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
    ```
* Graph Edit Distance (GED):
    * `graph-matching-toolkit`
        * `cd src && git clone https://github.com/yunshengb/graph-matching-toolkit.git`
        * Follow the instructions on https://github.com/yunshengb/graph-matching-toolkit
    * `java`
    