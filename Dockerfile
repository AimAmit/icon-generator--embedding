FROM public.ecr.aws/lambda/python:3.11

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu sentence-transformers
RUN pip3 install pinecone-client

COPY all-MiniLM-L6-v2 all-MiniLM-L6-v2

COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.lambda_handler" ]
