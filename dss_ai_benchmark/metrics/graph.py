from matplotlib import pyplot as plt
from abc import abstractmethod

class Graph(object):
    def __init__(self,**kwargs):
        self.data = kwargs["data"]
        self.path = kwargs["path"]
        self.logger = kwargs["logger"]
        self.plt = plt


    def process_data(self):
        pass

    @abstractmethod
    def draw(self):
        self.logger.error("No Implementation!")

    def save(self):
        """
        Save the image into a specified path
        :return:
        """
        graph_file_path = self.path + "/dss_dnn_bench_graph"
        try:
            self.plt.savefig(graph_file_path)
            self.logger.info("Saved Graph at {}.png".format(graph_file_path))
        except IOError as e:
            self.logger.error(e)




class SampleGraph(Graph):

    def __init__(self,**kwargs):
        super(SampleGraph, self).__init__(data=kwargs["data"],
                                          path=kwargs["path"],
                                          logger=kwargs["logger"])
    def draw(self):
        """
        Draw graphs
        :return:
        """
        if not self.data:
            self.logger.error("Graph data is not available.!")
            return
        self.logger.info("Drawing the Graphs!")
        X_label = self.data[0].pop(0)
        X_data  = self.data[0]
        #sub_plot_count = len(self.data) -1
        #figure,sub_plot = self.plt.subplots(sub_plot_count,1)

        # BatchSize vs BW
        #index = 1
        #while index < len(self.data):
        #    Y_label = self.data[index].pop(0)
        #    print(X_data, self.data[index])
        #    sub_plot[index,0].plot(X_data, self.data[index])
        #    sub_plot[index, 0].set_title("{} vs {}".format(Y_label,X_label))
        #    index +=1
        Y_label = self.data[3].pop(0)
        self.plt.plot(X_data,self.data[3], label=Y_label )
        self.plt.legend()
        #self.plt.show()

class CustomGraph(object):
    def __init__(self,name,data,path,logger):
        self.data = data
        assert(self.data is not None)
        self.name = name
        self.path = path
        self.logger = logger
        self.class_name = self.get_class_name()

    def get_class_name(self):
        """
        Convert the string class name to actual class
        :return:
        """
        try:
            return eval(self.name) # Convert the string to class name.
        except NameError as e:
            self.logger.execp("ERROR: Custom dataset doesn't exist! {}".format(e))
            sys.exit()

    def get_graph(self):
        self.logger.info("INFO: Using GraphClass - {}->{}".format(self.name,self.class_name))
        return self.class_name(data=self.data,
                               path=self.path,
                               logger=self.logger)

