import pandas as pd
import time
import os
import datetime

class History(object):
    """
    Authors : Alix Leroy,
    The class stores and manages the history
    """

    def __init__(self,
                 metrics,
                 num_epochs,
                 num_batches,
                 initial_epoch = "auto",
                 log_dir = "../../results/",
                 file_name = "history.csv",
                 verbose=1,
                 save_condition = "end_training", # "end_training" to save at the end of training, "end_epoch" to save at the end of the epoch,
                 ):


        self.verbose = verbose
        self.save_condition = save_condition

        self.running_metrics = {}

        self.num_current_batch = 1
        self.num_current_epoch = initial_epoch

        self.num_total_epochs = num_epochs
        self.num_total_batches = num_batches


        self.metrics = metrics
        self.history = pd.DataFrame(columns=["wall time", "relative time", "epoch"] + metrics)
        self.start_time = 0

        self.file_path = "%s/%s" % (log_dir, file_name)
        self.__load_history()
        self.__set_initial_epoch()

    def on_train_begin(self):
        """
        Authors : Samuel Westlake, Alix Leroy
        Called at the begining of the training
        :return: None
        """
        self.__set_start_time()


    def on_batch_end(self, metrics):
        """
        Called at the end of every batch
        Author : Alix Leroy
        :param losses:
        :param metrics:
        :return:
        """

        # Save the data in memory
        self.running_metrics += metrics.item()

        # If verbose == 2 print the statistics
        if self.verbose == 2:

            print_metrics = " ,".join([str(m[0]) + " : " +  str(m[1]) for m in metrics])


            print("Batch " + str(self.num_current_batch) + "/" + str(self.num_total_batches) + " : Metrics : " + str(print_metrics))

        # Get ready for the next batch
        self.num_current_batch += 1


    def on_epoch_end(self):
        """
        Authors : Alix Leroy,
        Called at the end of every epoch of the training
        :return: None
        """

        # If we want to display the metrics at the end of each epoch
        if self.verbose >= 1:

            print_metrics = " ,".join([str(m[0]) + " : " +  str(m[1] / self.num_total_batches) for m in self.running_metrics])


            print("\n")
            print("==============================================================================================================================")
            print("Epoch " + str(self.num_current_epoch) + "/" + str(self.num_total_epochs) + " : Metrics : "  + str(print_metrics))
            print("==============================================================================================================================")
            print("\n")




        # Save the history in memory
        data = dict(list(zip(self.metrics, [self.running_metrics.get(m) for m in self.metrics])))
        data["wall time"] = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
        data["relative time"] = self.__time()
        data["epoch"] = self.num_current_epoch
        self.history = self.history.append(data, ignore_index=True)


        # Reset the number of the current batch
        self.num_current_batch = 0

        # Get ready for the next epoch
        self.num_current_epoch +=1

        # Save the history
        self.__save_history()


    def on_training_end(self):
        """
        Authors: Alix Leroy
        Actions to perform when the training finishes
        :return: None
        """


        self.__save_history()


        print("\n")
        print("==============")
        print("HISTORY SAVED")
        print("==============")
        print("\n")


    def __save_history(self):
        """
        Authors: Alix Leroy
        Save the history into a CSV file
        :return: None
        """

        self.history.to_csv(self.file_path, index=True)


    def __load_history(self):
        """
        Load the existing history in memory
        :return:
        """
        if os.path.isfile(self.file_path):
            self.history = pd.read_csv(self.file_path)



    def __set_start_time(self):


        # If we did not train the model before we start the time at NOW
        if self.history.empty:
            self.start_time = time.time()


        # Else use the last time of the history
        else:
            self.start_time = time.time() - self.history["relative time"][self.history.index[-1]]


    def __set_initial_epoch(self):
        """
        Authors : Samuel Westlake, Alix Leroy
        Set the initial epoch of the training
        :return: None
        """

        #
        if self.num_current_epoch == "auto":
            if self.history.empty:
                self.num_current_epoch = 0
            else:
                self.num_current_epoch = self.history["epoch"][self.history.index[-1]]


        elif isinstance(self.num_current_batch, int) is False:
            raise ValueError("The given initial epoch is neither a integer nor on auto mode")


    def __time(self):
        """
        :return: Current train time in seconds
        """
        return round(time.time() - self.start_time, 2)

