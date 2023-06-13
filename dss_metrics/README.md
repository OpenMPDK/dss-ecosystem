
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
python3 metrics.py --config <path to config.json>
```

With filter
```
python3 metrics.py --filter --config <path to config.json>
```

## Architecture
Metrics Agent runs on a per node basis. A custom collector is used to collect from each subsystem (i.e. Minio, Target, etc..). These 
collector are run in parallel using Multiprocess and store information of the metrics into a shared buffer. Some of this information 
includes unique identifiers to assist with data analysis / correlation later, such as; minio_id, target_id, subsystem_id, timestamp, etc. 
The collected metric information later gets converted into actual Prometheus Metric objects and then "exported" or yielded to an endpoint 
that can be consumed by users

## Prometheus DB
Prometheus must be run locally from the node. Prometheus will continuosuly scrape the endpoint at which metrics are exported to and store 
in the local prometheus DB. The configuration for DSS Metrics Agent for Prometheus can be found in the "config/prometheus.yml" file. 
In order to start prometheus run the following from where prometheus is installed and specify the DSS Metrics Agent prometheus.yml file;
```
./prometheus --config.file=<path to prometheus.yml>
```

## Resources
Refer to Prometheus Getting Started section for download, installation, and execution of Prometheus
```
https://prometheus.io/docs/prometheus/latest/getting_started/
```
