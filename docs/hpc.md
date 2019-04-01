# Using Deeplodocus on a HPC

## Using without installing

If you don't have admin right, so cannot install Deeplodocus on your machine, you can still use deeplodocus by downloading and running the source code. 

1. If you have git installed, simply run: 

    ```text
    git clone https://github.com/Deeplodocus/deeplodocus.git
    ```

    Otherwise, head over to the [Deeplodocus repository](https://github.com/Deeplodocus/deeplodocus) and click on `Clone or download`, `Download ZIP` and extract the contents to your prefered location. 

2. Next, set the path to deeplodocus (Explanation to come!)

3. Finally, run:

    ```text
    python3 deeplodocus/core/project/project_utility.py
    ```
    
    which will create a blank Deeplodocus project in the top-leve `test` directory. 

## On wake
When using Deeplodocus on a HPC, you will likely not have access to the Deeplodocus terminal while your job is running.
Therefore, you will have to use `on_wake` commands, what can be specified in `config/project.yaml`

As the name suggests, on wake commands are run when Deeplodocus starts up thus, your on wake configurations for a simple training job may look like this:

```yaml
on_wake:
  - load()
  - train()
  - sleep()
``` 

