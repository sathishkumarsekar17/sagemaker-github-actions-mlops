import os
import json
import boto3
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["S3_BUCKET"]
session = PipelineSession(
    boto_session=boto3.Session(region_name=region),
    default_bucket=bucket,
)

processor = SKLearnProcessor(
    framework_version="1.2-1", role=role,
    instance_type="ml.m5.large", instance_count=1,
    sagemaker_session=session,
)
process_args = processor.run(
    code="src/preprocess.py",
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
)
preprocess = ProcessingStep(name="PreprocessData", step_args=process_args)

def output_uri(name):
    return preprocess.properties.ProcessingOutputConfig.Outputs[name].S3Output.S3Uri

estimator = SKLearn(
    entry_point="train.py", source_dir="src",
    framework_version="1.2-1", py_version="py3",
    role=role, instance_type="ml.m5.large", instance_count=1,
    sagemaker_session=session,
)
train_args = estimator.fit({
    "train": TrainingInput(s3_data=output_uri("train")),
    "test": TrainingInput(s3_data=output_uri("test")),
})
training = TrainingStep(name="TrainModel", step_args=train_args)

model = SKLearnModel(
    model_data=training.properties.ModelArtifacts.S3ModelArtifacts,
    role=role, entry_point="inference.py", source_dir="src",
    framework_version="1.2-1", py_version="py3",
    sagemaker_session=session,
)
register_args = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="CancerClassificationModels",
    approval_status="Approved",
)
registration = ModelStep(name="RegisterModel", step_args=register_args)

pipeline = Pipeline(
    name="CancerClassificationPipeline",
    steps=[preprocess, training, registration],
    sagemaker_session=session,
)
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Execution ARN:", execution.arn, flush=True)
with open("execution.json", "w") as file:
    json.dump({"execution_arn": execution.arn}, file)
execution.wait()
status = execution.describe()["PipelineExecutionStatus"]
if status != "Succeeded":
    raise RuntimeError(f"Pipeline failed: {status}")
print("Pipeline succeeded")