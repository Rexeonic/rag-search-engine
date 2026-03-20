keyword-search-cli is a python script.

has 4 commmands:

1. search

    SYNOPSIS
        keyword-search-cli search Query

        eg. 
            keyword-search-cli search "the mission movie with protocol"

            1. Mission Impossible: Ghost Protocol
            .
            .
            .

2. build

    SYNOPSIS
        keyword-search-cli build

    Description
        build cmd is builds the cache. It uses the 
        'cli.lib.preprocessing' to build cache files

            * index.pkl
            * docmap.pkl
            * term_frequencies.pkl