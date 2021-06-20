from azureml.core import Dataset
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.datastore import Datastore
from azureml.core.workspace import Workspace

ws = Workspace.from_config()
# cluster_name = "ML-CPU-test"
cluster_name = "ML-GPU-test"
compute_target = None

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('No compute cluster named {}'.format(cluster_name))
    exit()

curated_env_name = 'Resnet50v15-CPU-cluster'
# pytorch_env = Environment.get(workspace=ws, name=curated_env_name)
pytorch_env = Environment.from_conda_specification(name=curated_env_name, file_path='./conda_dependencies.yml')

project_folder = './'
data_path = 'datasets'

datastore = Datastore.get(ws, 'workspaceblobstore')
dataset = Dataset.File.from_files(path=(datastore, data_path))
data_loc = dataset.as_named_input('input').as_mount()

# src = ScriptRunConfig(source_directory=project_folder,
#                       # command=['ls'],
#                       script='train_resnet.py',
#                       arguments=[
#                           '--num_epochs', 16,
#                           '--batch', '32',
#                           '--shuffle', 'True',
#                           '--dataloc', data_loc,
#                           '--output_dir', './outputs',
#                       ],
#                       compute_target=compute_target,
#                       environment=pytorch_env)
#
# run = Experiment(ws, name='Train-Resnet50v15').submit(src)

src = ScriptRunConfig(source_directory=project_folder,
                      # command=['ls'],
                      script='cifar10-test.py',
                      arguments=[
                          '--num_epochs', 70,
                          '--batch', '128',
                          '--shuffle', 'True',
                          '--dataloc', data_loc,
                          '--output_dir', './outputs',
                      ],
                      compute_target=compute_target,
                      environment=pytorch_env)

run = Experiment(ws, name='Resnet50v15-cifar10').submit(src)
print("Script submitted")

run.wait_for_completion(show_output=True)
