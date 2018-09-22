from .callback import Callback

from torch.utils.data import  DataLoader


class Trainer(object):

    def __init__(self, model, dataset, metrics, losses, optimizer, epochs, initial_epoch, batch_size = 4, shuffle = True, num_workers = 4):

        self.model = model
        self.metrics = metrics
        self.losses = losses
        self.optimizer = optimizer
        self.callbacks = Callback()
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    def start(self, method):
        """
        :param method:
        :return:
        """

        if method == "from_directory":
            self.__train_from_directory()

        elif method == "from_file":
            self.__train_from_file(metrics, losses, optimizer, callbacks)

        else:
            raise ValueError("Unknown training method %s" % method)


        print("Training...")


    def __train_from_file(self, first_training = True):


        if first_training is True :

            self.callbacks.on_train_begin()


        for epoch in range(self.initial_epoch, self.epochs):  # loop over the dataset multiple times


            for i, batch in enumerate(self.dataloader, 0):

                # get the inputs
                inputs, labels, addtional_data = batch

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)          #infer the outputs of the network

                criterion = self.losses[0]

                loss = criterion(inputs, outputs, labels, additional_data)   # Compute the loss
                metrics = __compute_metrics()

                loss.backward() # accumulates the gradient (by addition) for each parameter
                self.optimizer.step() # performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule

                # Minibatch callback
                self.callbacks.on_batch_end(losses, metrics)


            #Epoch callback
            self.callbacks.on_epoch_end(um_total_epochs=num_total_epochs)

        # End of training callback
        self.callbacks.on_training_end()


        print("\n")
        print("=============================================================")
        print('Finished Training')
        print("=============================================================")
        print("\n")


        self.__continue_training()


    def __continue_training(self):

        continue_training = ""

        # Ask if the user want to continue the training
        while continue_training != ("Y" or "y" or "N" or "n")

            continue_training = input('Would you like to continue the training ? (Y/N) ')

        #If yes ask the number of epochs
        if continue_training == ("Y" or "y"):
            epochs = ""

            while not isinstance(epochs, int):
                epochs = input('Number of epochs ? ')


        # Reset the system to continue the training
        if epochs > 0:
            self.initial_epoch = self.epochs
            self.epochs += epochs

            # Resume the training
            self.__train_from_file(first_training = False)







    def __train_from_directory2(self):
        """
        Author: Samuel Westlake
        :return: None
        """
        # Initialize data generators
        train_kwargs = {**self.config["data_gen"]["common"], **self.config["data_gen"]["train"]}
        val_kwargs = {**self.config["data_gen"]["common"], **self.config["data_gen"]["val"]}
        train_gen = DataGenerator(**train_kwargs)
        val_gen = DataGenerator(**val_kwargs)

        # Set the initial epoch in fit_gen
        fit_gen_kwargs = copy.deepcopy(self.config["fit_gen"])
        fit_gen_kwargs = self.__set_fit_epoch(fit_gen_kwargs)
        fit_gen_kwargs = self.__set_fit_class_weight(fit_gen_kwargs, train_gen)

        callbacks = self.__initialize_callbacks()

        self.net.fit_generator(train_gen.flow(),
                                 steps_per_epoch=train_gen.steps(),
                                 validation_data=val_gen.flow(),
                                 validation_steps=val_gen.steps(),
                                 callbacks=callbacks,
                                 **fit_gen_kwargs)

    def __train_from_file2(self):
        # Combine common arguments with train/validation arguments
        train_args = {**self.config["data_gen"]["common"], **self.config["data_gen"]["train"]}
        validation_args = {**self.config["data_gen"]["common"], **self.config["data_gen"]["val"]}

        # Initialise the generators
        train_generator =DataGenerator(**train_args)
        train_generator.init_flow(self.config["data_gen"]["train"]["x_files"],
                                  self.config["data_gen"]["train"]["y_files"],
                                  self.config["data_gen"]["common"]["x_types"],
                                  self.config["data_gen"]["common"]["y_types"])

        val_generator = DataGenerator(**validation_args)
        val_generator.init_flow(self.config["data_gen"]["val"]["x_files"],
                                self.config["data_gen"]["val"]["y_files"],
                                self.config["data_gen"]["common"]["x_types"],
                                self.config["data_gen"]["common"]["y_types"])

        # Get length of the data
        n_train = train_generator.samples()
        n_val = val_generator.samples()
        #n_test = test_generator.samples()
        # Create the generators
        generator = train_generator.flow()
        validation_data = val_generator.flow()

        # Calculate the number of steps in one epoch for training and validation
        steps_per_epoch = int(np.ceil(n_train / self.config["data_gen"]["common"]["batch_size"]))
        validation_steps = int(np.ceil(n_val / self.config["data_gen"]["common"]["batch_size"]))

        # Initialise callbacks
        callbacks = self.__initialize_callbacks()

        # Set the initial epoch in fit_gen
        fit_gen_kwargs = copy.deepcopy(self.config["fit_gen"])
        fit_gen_kwargs = self.__set_fit_epoch(fit_gen_kwargs)
        #fit_gen_kwargs = self.__set_fit_class_weight(fit_gen_kwargs, train_gen)

        print("Currently training using the flow() function ...")
        self.net.fit_generator(generator,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks,
                                 **fit_gen_kwargs)
