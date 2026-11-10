import os
import json
import time
import boto3

sm = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])
execution_arn = json.load(open("execution.json"))["execution_arn"]
steps = sm.list_pipeline_execution_steps(
    PipelineExecutionArn=execution_arn
)["PipelineExecutionSteps"]
registered = next(s for s in steps if s["StepName"] == "RegisterModel")
package_arn = registered["Metadata"]["RegisterModel"]["Arn"]

stamp = str(int(time.time()))
model_name = f"cancer-model-{stamp}"
config_name = f"cancer-config-{stamp}"
endpoint_name = "cancer-classification-endpoint"

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
    Containers=[{"ModelPackageName": package_arn}],
)
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
        "InitialVariantWeight": 1,
    }],
)
try:
    sm.describe_endpoint(EndpointName=endpoint_name)
except sm.exceptions.ClientError as e:
    if e.response["Error"]["Code"] != "ValidationException":
        raise
    sm.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=config_name
    )
else:
    sm.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=config_name
    )
sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
print("Endpoint InService:", endpoint_name)