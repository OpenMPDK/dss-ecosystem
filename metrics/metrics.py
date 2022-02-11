import os
from metrics.graph import CustomGraph


class Metrics(object):

    def __init__(self,config,data,logger):
        self.config = config
        self.data = data
        self.logger = logger
        self.path = self.config["path"]

        # Graph enabled
        self.graph_config = self.config["graph"]
        self.graph = None
        self.graph_data = []

    def process(self):
        print(self.data)
        self.save()
        # Check graph plotting is enabled?
        if self.graph_config["enabled"]:
            self.graph_plot()

    def save(self):
        """
        Save data to persistent storage
        :return:
        """

        self.metrics_file_name = self.path + "/metrics.csv"
        try:
            mode = "w"
            if os.path.exists(self.metrics_file_name) and os.path.isfile(self.metrics_file_name):
                mode = "a"
                self.data.pop(0)

            with open(self.metrics_file_name,mode) as fh:
                for record in self.data:
                    fh.write( ",".join(record) + "\n")
            self.logger.info("Metrics data stored at {}".format(self.metrics_file_name))
        except IOError as e:
            self.logger.error(e)
        except Exception as e:
            self.logger.error("{}".format(e))


    def graph_plot(self):
        """
        PLot graph and store into a file
        :return:
        """

        record_fields_count = len(self.data[0])
        is_header = True
        with open(self.metrics_file_name, "r") as fh:
            for record in fh.readlines():
                record = record.strip()
                fields = record.split(",")
                if is_header:
                    for field in fields:
                        self.graph_data.append([field])
                    is_header = False
                else:
                    for index in range(record_fields_count):
                        self.graph_data[index].append(fields[index])

        graph_class_name = self.graph_config["name"]
        cg = CustomGraph(name=graph_class_name,
                         data=self.graph_data,
                         path=self.path,
                         logger=self.logger)
        self.graph = cg.get_graph()
        self.graph.draw()
        self.graph.save()

