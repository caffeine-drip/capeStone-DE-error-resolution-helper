# save as generate_spark_errors.py
import os, json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

client = OpenAI()

class SparkError(BaseModel):
    id: str
    error_name: str
    error_message: str = Field(description="Actual exception text users see")
    component: str = Field(description="driver/executor/shuffle/yarn/hdfs/etc")
    cause: str
    fix: str = Field(description="Step by step resolution")
    prevention: str

class SparkErrorDataset(BaseModel):
    errors: List[SparkError]

prompt = """Generate 50 real Apache Spark error records covering:
- OOM errors (driver and executor)
- Shuffle FetchFailed exceptions
- Stage failures and task failures
- HDFS/S3 connectivity errors
- Schema/serialization errors
- YARN resource errors
- Broadcast timeout errors
- Checkpoint errors
- PySpark Python worker errors
Each must have a real-looking exception message and a concrete fix."""

response = client.beta.chat.completions.parse(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    response_format=SparkErrorDataset,
)

errors = [e.model_dump() for e in response.choices[0].message.parsed.errors]
print(f"Generated {len(errors)} errors")
with open("data/generated_spark_errors.json", "w") as f:
    json.dump(errors, f, indent=2)