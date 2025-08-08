#!/bin/bash

# FTP服务器信息
HOST="ftp2.tpdc.ac.cn"  # 或 ftp3.tpdc.ac.cn
PORT="6201"
USER="download_38116006"
PASS="51117084"

# 需要下载的远程目录
REMOTE_DIR_1="ftp://$HOST:$PORT/NNsm_nc/2021/"
REMOTE_DIR_2="ftp://$HOST:$PORT/NNsm_nc/2022/"

# 本地保存的目标目录
LOCAL_DIR="./downloaded_data"

# 创建本地目录
mkdir -p $LOCAL_DIR

# 使用 wget 下载第一个目录
wget --ftp-user="$USER" --ftp-password="$PASS" -r -nH --cut-dirs=2 -P "$LOCAL_DIR" "$REMOTE_DIR_1"

# 使用 wget 下载第二个目录
wget --ftp-user="$USER" --ftp-password="$PASS" -r -nH --cut-dirs=2 -P "$LOCAL_DIR" "$REMOTE_DIR_2"
