#!/bin/bash

SIZE_LR=1
HS1=150
HL1=3
WS=9
SE=5
BIDIR1=1
NORM=ln

BATCH=10
#FEAT1=SN_SF1_SIN_SF1_F6
FEAT1=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM1=9

START_EPOCH=0
END_EPOCH=80

LOG_FILE="log/sdt6_resnet_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample.log"

mkdir -p log

echo -e "Saving Record to ${LOG_FILE}"

bash script/train_TONAS_sdt6_resnet.sh ${SIZE_LR} ${HS1} ${HL1} ${WS} ${SE} ${BIDIR1} ${NORM} ${BATCH} ${FEAT1} ${FEAT_NUM1} ${START_EPOCH} ${END_EPOCH} | tee ${LOG_FILE}

echo -e "Saving Record to ${LOG_FILE}"