#!/usr/bin/sh

batch_size=50
workers=50
while [ $workers -le 100 ]; do
  #echo Executing with workers=$workers
  while [ $batch_size -le 100 ]; do
  echo Executing with workers=$workers, batch_size=$batch_size
  python3 benchmark.py  --dataloader_workers $workers --batch_size $batch_size
  batch_size=$(( $batch_size + 20 ))
  done
  batch_size=50
  workers=$(( $workers + 20 ))
done
