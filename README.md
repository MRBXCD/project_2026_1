# A Humor Generation Model

## 1 Introduction

## 2 Installation / Environment

### Key Dependence
The `flash-attn` pkg may cause some issue, use the following command to download pre-built wheel:
```shell
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
Otherwise it may takes an hour to build the flash attention from scratch. Make sure to choose correct version corresponding to a specific python and cuda version

After installing the `flash-attn`, execute the following command to install other dependence:
```shell
pip install -r requirements.txt
```