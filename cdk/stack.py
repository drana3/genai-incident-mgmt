from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_sqs as sqs,
    aws_events as events,
    aws_events_targets as targets,
    aws_dynamodb as dynamodb,
    aws_opensearchservice as opensearch,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ssm as ssm,
    aws_sns as sns,
    aws_stepfunctions as sfn,
    aws_apigateway as apigw,
    aws_aps as aps,
    aws_xray as xray,
    Duration
)
from aws_cdk.aws_ec2 import EbsDeviceVolumeType as EbsVolumeType  # Correct import
from constructs import Construct


class IncidentMgmtStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(self, "ArtifactsBucket", versioned=True,
                           encryption=s3.BucketEncryption.S3_MANAGED)

        audit_table = dynamodb.Table(self, "IncidentAudit",
                                     partition_key=dynamodb.Attribute(
                                         name="incident_id", type=dynamodb.AttributeType.STRING),
                                     sort_key=dynamodb.Attribute(
                                         name="timestamp", type=dynamodb.AttributeType.STRING),
                                     billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                                     encryption=dynamodb.TableEncryption.AWS_MANAGED
                                     )

        opensearch_domain = opensearch.Domain(self, "RunbooksDomain",
                                              version=opensearch.EngineVersion.OPENSEARCH_2_11,
                                              capacity=opensearch.CapacityConfig(
                                                  data_nodes=3,
                                                  data_node_instance_type="r6g.large.search",
                                                  multi_az_with_standby_enabled=True
                                              ),
                                              ebs=opensearch.EbsOptions(
                                                  volume_size=20, volume_type=EbsVolumeType.GP3),
                                              encryption_at_rest=opensearch.EncryptionAtRestOptions(
                                                  enabled=True),
                                              node_to_node_encryption=True
                                              )

        process_lambda = _lambda.Function(self, "ProcessLambda",
                                          runtime=_lambda.Runtime.PYTHON_3_12,
                                          handler="main.handler",
                                          code=_lambda.Code.from_asset(
                                              "../app"),
                                          timeout=Duration.minutes(5),
                                          memory_size=512,
                                          environment={
                                              "OPENSEARCH_URL": f"https://{opensearch_domain.domain_endpoint}",
                                              "AUDIT_TABLE": audit_table.table_name
                                          },
                                          tracing=_lambda.Tracing.ACTIVE
                                          )
        audit_table.grant_read_write_data(process_lambda)
        opensearch_domain.grant_read_write(process_lambda)
        process_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "ssm:SendCommand",
                     "ssm:GetCommandInvocation", "sns:Publish", "opensearch:ESHttp*"],
            resources=["*"]
        ))

        api = apigw.LambdaRestApi(
            self, "ApiGateway", handler=process_lambda, proxy=True)

        queue = sqs.Queue(self, "AlertQueue", dead_letter_queue=sqs.DeadLetterQueue(
            max_receive_count=3, queue=sqs.Queue(self, "AlertDLQ")))
        process_lambda.add_event_source_mapping(
            "SQSMapping", event_source_arn=queue.queue_arn, batch_size=10)
        queue.grant_consume_messages(process_lambda)

        rule = events.Rule(self, "AlertRule",
                           event_pattern=events.EventPattern(
                               source=["aws.cloudwatch"])
                           )
        rule.add_target(targets.SqsQueue(queue))

        notify_topic = sns.Topic(self, "NotifyTopic")

        workflow = sfn.StateMachine(self, "HumanWorkflow",
                                    definition=sfn.Chain.start(
                                        sfn.Task(self, "NotifyHuman",
                                                 resource="arn:aws:states:::sns:publish",
                                                 parameters={"Message.$": "$.resolution",
                                                             "TopicArn": notify_topic.topic_arn}
                                                 ).next(
                                            sfn.Wait(self, "WaitForApproval",
                                                     time=sfn.WaitTime.duration(Duration.minutes(5)))
                                        ).next(
                                            sfn.Task(self, "ExecuteAfterApprove",
                                                     resource=process_lambda.current_version.function_arn)
                                        )
                                    )
                                    )
        notify_topic.grant_publish(workflow)
        process_lambda.grant_invoke(workflow)

        amp_workspace = aps.CfnWorkspace(self, "MonitoringWorkspace")
