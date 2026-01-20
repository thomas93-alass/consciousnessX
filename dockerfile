FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -e .[all]
EXPOSE 8050
CMD ["python", "src/visualization/consciousness_dashboard.py"]
