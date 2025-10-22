# Create CCPG Using Joern
```
# ccpg without line number
cmd = f"joern --script extract_func.sc --param codeDir={codeDir} --param outputDir={outputDir}"
# ccpg with line number 
use convert2ccpg.sc

```
### Parameters

| Parameter  | Description                                                                 | Required |
|------------|-----------------------------------------------------------------------------|----------|
| `codeDir`  | Directory containing a C/C++ source file to analyze                          | Yes      |
| `outputDir` | Directory where output files will be saved                                  | Yes      |
| `format`   | Output format: `ccpg` , or `cpg` (origin CPG) | Yes      |

## Example
```
cd joern-parse-ccpg/
joern --script convert2ccpg.sc --param codeDir=/home/fzc/workspace/joern-parse/joern-export-demo/c_code --param outputDir=/home/fzc/workspace/joern-parse/joern-export-demo/out1 --param format=ccpg
```

# data process

```
cd ./preprocess/deeprace_preprocess
# First, run normal.p yand normal1.py to preprocess comments and special lines.
# Second,code to dot
python code_to_con_dot.py
# dot to json
python dot_to_json.py
```
use ./preprocess/test.ipynb to generate train, valid, test.txt containing file paths.
train word2vec model

```
python train_w2v.py
```

or train doc2vec model

```
python train_d2v.py
```

or train codebert model
```
python train_codebert2v.py
```


# vul_detect
in directory ./vul_detect

# subgraphx
in directory ./SubgraphX