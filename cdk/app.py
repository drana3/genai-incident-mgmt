#!/usr/bin/env python3
from aws_cdk import App
from stack import IncidentMgmtStack

app = App()
IncidentMgmtStack(app, "GenAIIncidentMgmtStack")
app.synth()
