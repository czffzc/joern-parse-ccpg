# Create CCPG Using Joern
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
