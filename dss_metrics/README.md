
# DSS Metrics Agent

## Overview
The DSS Metrics Agent runs on a per node level. It can collect data from MINIO, target, and other sources. The data is then processed
into metrics that can be stored into a Promotheus database or exposed at an endpoint. The metrics are associated with tags such as 
cluster_id, susbsystem_id, target_id, etc.. in order to be able to correlate the metric with a given point/layer in the system. It 
is assumed that all captured metrics will be time series data.

## Execution

Until the agent is deployed along with the deploy DSS playbook, it will need to be manually run. Currently, the agent supports
exporting all available metrics from the system or a smaller subset of metrics as defined in the 'whitelist.txt' file.

Without filter
```
python3 metrics.py
```

With filter
```
python3 metrics.py --filter
```