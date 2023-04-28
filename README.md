# SVD and HITS Recommender Experiments

## How to Run All Experiments
1. Install requirements for compiling Python 3.10 (change configure-dep if package manager is not apt)

```sudo ./configure-dep```

or create file to skip this step

```touch configure-dep.completed```

2. Install Python 3.10 locally and configure environment (edit configure script experiment values before running to change)

```. ./configure```

(re-run to reconfigure environment, only installs Python once)

(if system Python version >= 3.10, possibly 3.9 (untested), set install_local_python=false in configure to avoid compiling/installing local python 3.10)

3. Add formatted data

add ```data.csv``` to ```data/```

data has format (all integer, rating can be float), use matching headers (order doesn't matter)

user,item,rating,time,{attributes, e.g. EC for entry college}  
0,0,3.3,0,5  
...

if no entry college data, remove ```configs/templates/attribute-template.json``` or update with your attribute

add ```filter_data.json``` to ```data/```, if desired

filter data has format (user: not_allowed_items)

```
{0: [1, 3, 7. 8, 10, ...],
 1: [...],
 ...
}
```

or

if no filter data, delete ```configs/templates/filtered_template.json```



5. Run Experiments, generate tables and figures

```make```

or run specific target (see makefile)

```make {target}```

## How to Run the Parameter Analysis Separately
```python3 __main__.py -n 30 -s 0 configs/parameter_analysis.json```

Runs the experiment using a minimum of 30 neighbors with seed 0 and configuration file ```configs/parameter_analysis.json```
