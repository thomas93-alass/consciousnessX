FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -e .[all]
CMD ["python", "examples/basic_demo.py"]
